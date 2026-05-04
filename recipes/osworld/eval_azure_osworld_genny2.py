"""OSWorld eval on Azure — Genny2 agent, configurable model and tool choice.

Uses AzureInfraConfig to launch fresh VMs per task.

Tool choices (OSWORLD_TOOL env var):
    "pyautogui"   — screenshot + accessibility tree → run_pyautogui() (default)
    "computer_13" — screenshot only → 13 mouse/keyboard primitives

Model (OSWORLD_MODEL env var), e.g.:
    "azure/gpt-5.4-mini"  (default)
    "azure/gpt-5.4"
    "azure/gpt-5.4-nano"

Prerequisites:
    See cube-resources/cube-infra-azure/README.md for full setup instructions.

Usage:
    # Debug mode (sequential)
    uv run recipes/osworld/eval_azure_osworld_genny2.py debug

    # Eval mode defaults (pyautogui tool, gpt-5.4-mini)
    uv run recipes/osworld/eval_azure_osworld_genny2.py

    # Different model / tool
    OSWORLD_MODEL=azure/gpt-5.4 OSWORLD_TOOL=computer_13 uv run recipes/osworld/eval_azure_osworld_genny2.py

    # Resume a prior run (model/tool/subset inferred from saved config)
    uv run recipes/osworld/eval_azure_osworld_genny2.py --resume ~/cube_harness_results/<run_dir>
"""

import logging
import os
import sys
from pathlib import Path

from cube_computer_tool.computer import ActionSpace
from cube_infra_azure import AzureInfraConfig
from dotenv import load_dotenv
from osworld_cube.benchmark import OSWorldBenchmarkConfig
from osworld_cube.computer import ComputerConfig

from cube_harness import make_experiment_output_dir
from cube_harness.agents.genny2 import Genny2Config
from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
for _noisy in ("azure.core.pipeline.policies.http_logging_policy", "azure.identity", "urllib3.connectionpool"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

INFRA = AzureInfraConfig(
    resource_group=os.environ.get("AZURE_RESOURCE_GROUP") or "ui_assist",
    storage_account=os.environ.get("AZURE_STORAGE_ACCOUNT") or "cubeexpvhd",
    vnet_name="vnet-westus2",
    nsg_name="osworld-nsg",
    image_name_suffix="-generalized",
)

# System prompt for Tool 1: pyautogui + accessibility tree
SYSTEM_PROMPT_PYAUTOGUI_AXTREE = """\
You are a desktop automation agent controlling a real Ubuntu computer.

## Observations
Each step you receive an element table listing interactive UI elements with columns:
index, tag, name, text, x, y, w, h

Where (x, y) is the top-left corner and (w, h) is the size of each element.
To click the center of element at row i: center_x = x + w//2, center_y = y + h//2

## Actions
You control the computer by calling run_pyautogui(code) with valid Python/pyautogui code.

### Common pyautogui commands
- pyautogui.click(x, y)                       — left-click at pixel coordinates
- pyautogui.rightClick(x, y)                  — right-click at pixel coordinates
- pyautogui.doubleClick(x, y)                 — double-click at pixel coordinates
- pyautogui.typewrite('text', interval=0.05)  — type text character by character
- pyautogui.hotkey('ctrl', 'c')               — press key combination
- pyautogui.press('enter')                    — press a single key
- pyautogui.scroll(x, y, clicks=-3)           — scroll (negative clicks = down)
- pyautogui.dragTo(x, y, button='left')       — drag to coordinates

### Ending the task
- Call fail() if the task CANNOT be completed (infeasible tasks)
- Call done() when the task is successfully COMPLETE

## Strategy
1. Study the element table carefully to find the element you need to interact with
2. Calculate center coordinates: center_x = x + w//2, center_y = y + h//2
3. If the task is clearly impossible, call fail() immediately
4. Prefer hotkey shortcuts over mouse clicks when practical
5. After completing the task, verify by checking the next observation then call done()
6. Do not loop — if an action has no effect after 2 attempts, try a different approach\
"""

# System prompt for Tool 2: screenshot only + 13-action space
# No need to list actions here — NativeToolAdapter injects the exact schemas via the API.
SYSTEM_PROMPT_COMPUTER_13 = """\
You are a desktop automation agent controlling a real Ubuntu computer.

## Observations
Each step you receive a screenshot of the current screen. There is no element table —
read the pixels and estimate target coordinates from visual cues (element edges,
surrounding layout). Coordinates are pixels measured from the screen's top-left (0, 0).

## Actions
You control the computer by calling discrete primitive actions.

### Mouse
- click(button="left", x=-1, y=-1, num_clicks=1)  — click at coords; -1 = use cursor pos
- double_click(x=-1, y=-1)                        — double-click at coords
- right_click(x=-1, y=-1)                         — right-click at coords
- mouse_down(button="left")                       — press and hold mouse button
- mouse_up(button="left")                         — release held mouse button
- move_to(x, y)                                   — move cursor without clicking
- drag_to(x, y)                                   — click-and-drag from cursor pos to (x, y)
- scroll(dx, dy)                                  — scroll wheel (positive dy = down)

### Keyboard
- typing(text)                                    — type literal text
- press(key)                                      — press+release a single key (e.g. "enter", "tab", "esc")
- key_down(key)                                   — press a key without releasing
- key_up(key)                                     — release a held key
- hotkey(keys)                                    — key combo "ctrl+c" or "ctrl+shift+t"

### Ending the task
- fail()  — call if the task CANNOT be completed (infeasible tasks)
- done()  — call when the task is successfully COMPLETE
- wait()  — pause briefly to let UI catch up

## Strategy
1. Look at the screenshot carefully and identify your target element by visual cues
2. Estimate target coordinates from element edges and surrounding layout
3. If the task is clearly impossible, call fail() immediately
4. Prefer hotkey shortcuts over mouse clicks when practical
5. After completing the task, verify by checking the next observation then call done()
6. Do not loop — if an action has no effect after 2 attempts, try a different approach\
"""


_TOOL_SYSTEM_PROMPTS = {
    "pyautogui": SYSTEM_PROMPT_PYAUTOGUI_AXTREE,
    "computer_13": SYSTEM_PROMPT_COMPUTER_13,
}

_TOOL_SLUGS = {
    "pyautogui": "axtree_pyautogui",
    "computer_13": "screenshot_13actions",
}

_MODEL_SLUGS = {
    "azure/gpt-5.4-mini": "gpt54mini",
    "azure/gpt-5.4": "gpt54",
    "azure/gpt-5.4-nano": "gpt54nano",
}


def _infer_resume_params(output_dir: Path) -> tuple[str, str, str]:
    """Read model, tool, subset from a prior run's experiment_config.json."""
    import json

    config_path = output_dir / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No experiment_config.json found in {output_dir}")
    cfg = json.loads(config_path.read_text())

    # model_name lives at agent_config.llm_config.model_name
    model = cfg["agent_config"]["llm_config"]["model_name"]

    # tool is encoded in the action_space field of benchmark_config.tool_config
    action_space = cfg["benchmark_config"]["tool_config"]["action_space"]
    tool = action_space

    # subset is stored as the named_subset field on benchmark_config
    subset = cfg["benchmark_config"].get("named_subset", "test_nogdrive")

    return model, tool, subset


def main(debug: bool, resume_dir: str | None = None) -> None:
    if resume_dir:
        output_dir = Path(resume_dir).expanduser().resolve()
        if not output_dir.is_dir():
            raise ValueError(f"--resume path does not exist: {output_dir}")
        model, tool, subset = _infer_resume_params(output_dir)
        print(f"RESUMING from {output_dir}")
        print(f"  model={model}  tool={tool}  subset={subset}")
        resuming = True
    else:
        model = os.environ.get("OSWORLD_MODEL", "azure/gpt-5.4-mini")
        tool = os.environ.get("OSWORLD_TOOL", "pyautogui")
        subset = os.environ.get("OSWORLD_SUBSET", "test_small")
        resuming = False

    if tool not in _TOOL_SYSTEM_PROMPTS:
        raise ValueError(f"OSWORLD_TOOL must be one of {list(_TOOL_SYSTEM_PROMPTS)}; got {tool!r}")

    tool_slug = _TOOL_SLUGS.get(tool, tool)
    model_slug = _MODEL_SLUGS.get(model, model.split("/")[-1].replace(".", ""))
    exp_name = f"osworld_genny2_{tool_slug}_{model_slug}"

    if not resuming:
        output_dir: Path = make_experiment_output_dir(exp_name, "osworld-cube")

    llm_config = LLMConfig(model_name=model, temperature=1.0, set_cache_control="auto")
    agent_config = Genny2Config(
        llm_config=llm_config,
        system_prompt=_TOOL_SYSTEM_PROMPTS[tool],
        max_actions=100,
        enable_summarize=False,
    )

    tool_config = ComputerConfig(
        action_space=ActionSpace(tool),
        require_a11y_tree=(tool == "pyautogui"),
        observe_after_action=True,
    )

    benchmark_config = OSWorldBenchmarkConfig(
        tool_config=tool_config,
        use_som=False,
    )
    OSWorldBenchmarkConfig.install()
    # Subset selectable via env var so we can iterate fast on test_small (39)
    # before scaling to test_nogdrive (360). Default to test_small. We always
    # filter out gdrive-dependent tasks (test_small ∩ test_nogdrive) since
    # those need credentials we don't reliably have wired up.
    benchmark_config = benchmark_config.named_subset(subset)
    if subset in ("test_small", "test_all"):
        nogdrive_ids = {
            tid
            for tid, tm in OSWorldBenchmarkConfig.task_metadata.items()
            if "test_nogdrive" in (getattr(tm, "test_sets", None) or [])
        }
        kept = [tid for tid in benchmark_config.task_ids or list(benchmark_config.task_metadata) if tid in nogdrive_ids]
        n_dropped = len(benchmark_config.task_ids or benchmark_config.task_metadata) - len(kept)
        benchmark_config.task_ids = kept
        if n_dropped:
            logging.info("Filtered out %d gdrive task(s) from %s", n_dropped, subset)
    logging.info(
        "OSWorld eval: subset=%s, %d tasks", subset, len(benchmark_config.task_ids or benchmark_config.task_metadata)
    )

    print("--- pre-run cleanup ---")
    pre_deleted = INFRA.cleanup_orphaned_resources()
    if pre_deleted:
        n = sum(len(v) for v in pre_deleted.values())
        print(f"Cleaned up {n} orphaned resource(s) from prior runs")

    # Pre-warm the Azure CLI token cache on disk before Ray workers spawn.
    # Each Ray worker is a fresh Python process that calls `az account
    # get-access-token` on first auth — at n_cpus=100+ that subprocess storm
    # collides on `~/.azure/msal_token_cache.bin` and ~all workers fail with
    # CredentialUnavailableError. By running `az` once here, the cache is
    # populated; worker `az` calls then read the warm cache and return fast
    # without hitting login.microsoftonline.com.
    print("--- pre-warm Azure CLI token cache ---")
    from cube_infra_azure.azure import _get_cached_cred

    cred = _get_cached_cred()
    tok = cred.get_token("https://management.azure.com/.default")
    print(f"Pre-warmed token, expires in {(tok.expires_on - __import__('time').time()) / 60:.1f}min")

    exp = Experiment(
        name=exp_name,
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark_config=benchmark_config,
        infra=INFRA,
        max_steps=100,
        resume=resuming,
    )

    try:
        if debug:
            print("\n" + "=" * 60)
            print("DEBUG MODE: Running sequentially on Azure")
            print("=" * 60)
            print(f"Output directory: {output_dir}")
            print(f"Model: {model}  Tool: {tool}")
            print(f"Infra: {INFRA.fingerprint()}")
            print("=" * 60 + "\n")
            run_sequentially(exp)
        else:
            print("\n" + "=" * 60)
            print("EVAL MODE: Running OSWorld on Azure with Genny2")
            print("=" * 60)
            print(f"Output directory: {output_dir}")
            print(f"Model: {model}  Tool: {tool}")
            print(f"Infra: {INFRA.fingerprint()}")
            print("Parallelism: n_cpus=50")
            print("=" * 60 + "\n")
            run_with_ray(exp, n_cpus=50, max_retry_rounds=1)
    finally:
        # Sweep any VMs orphaned by Ray force-kills or worker crashes.
        # Normal completions are already cleaned up by task.close() in episode.py.
        deleted = INFRA.cleanup_orphaned_resources()
        if deleted:
            print(f"Cleaned up {len(deleted)} orphaned VM(s): {deleted}")


if __name__ == "__main__":
    args = sys.argv[1:]
    _debug = "debug" in args
    _resume_dir: str | None = None
    if "--resume" in args:
        idx = args.index("--resume")
        if idx + 1 >= len(args):
            print("Error: --resume requires a directory argument", file=sys.stderr)
            sys.exit(1)
        _resume_dir = args[idx + 1]
    main(_debug, resume_dir=_resume_dir)
