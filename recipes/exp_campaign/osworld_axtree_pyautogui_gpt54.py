"""OSWorld × Tool 1 (Screenshot+Axtree → pyautogui) × GPT-5.4 (Azure)."""

import os

# Drop unsupported params silently (some Azure models reject `tool_choice`).
os.environ.setdefault("LITELLM_DROP_PARAMS", "true")

import logging
import sys
from pathlib import Path

from cube_infra_azure import AzureInfraConfig
from dotenv import load_dotenv
from osworld_cube.benchmark import OSWorldBenchmarkConfig
from osworld_cube.computer import ComputerConfig

from cube_harness import make_experiment_output_dir
from cube_harness.agents.genny import GennyConfig
from cube_harness.exp_runner import run_with_ray
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig

import litellm  # noqa: E402
litellm.drop_params = True

sys.path.insert(0, str(Path(__file__).parent))
from _prompts import OSWORLD_TOOL1_AXTREE_PYAUTOGUI  # noqa: E402

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

MODEL_NAME = "azure/gpt-5.4"
EXP_NAME = "osworld_axtree_pyautogui_gpt54"
SUBSET = os.environ.get("OSWORLD_SUBSET", "test_nogdrive")


def main() -> None:
    resume_from = os.environ.get("OSWORLD_RESUME_DIR")
    if resume_from:
        output_dir = Path(resume_from)
        print(f"RESUMING from {output_dir}")
    else:
        output_dir = make_experiment_output_dir(EXP_NAME, "osworld-cube")

    system_prompt = OSWORLD_TOOL1_AXTREE_PYAUTOGUI

    llm_config = LLMConfig(model_name=MODEL_NAME, temperature=1.0)
    agent_config = GennyConfig(
        llm_config=llm_config,
        system_prompt=system_prompt,
        max_actions=100,
        render_last_n_obs=3,
        enable_summarize=False,
        tools_as_text=False,
    )

    tool_config = ComputerConfig(
        action_space="pyautogui",
        require_a11y_tree=True,
        observe_after_action=True,
    )

    benchmark_config = OSWorldBenchmarkConfig(tool_config=tool_config, use_som=False)
    OSWorldBenchmarkConfig.install()
    benchmark_config = benchmark_config.named_subset(SUBSET)
    logging.info("OSWorld eval: subset=%s, %d tasks", SUBSET, len(benchmark_config.task_ids or benchmark_config.task_metadata))

    print("--- pre-run cleanup ---")
    pre = INFRA.cleanup_orphaned_resources()
    if pre and any(pre.values()):
        n = sum(len(v) for v in pre.values())
        print(f"Cleaned up {n} orphaned resources from prior runs")

    print("--- pre-warm Azure CLI token cache ---")
    from cube_infra_azure.azure import _get_cached_cred
    cred = _get_cached_cred()
    tok = cred.get_token("https://management.azure.com/.default")
    print(f"Pre-warmed token, expires in {(tok.expires_on - __import__('time').time())/60:.1f}min")

    exp = Experiment(
        name=EXP_NAME,
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark_config=benchmark_config,
        infra=INFRA,
        max_steps=100,
        resume=bool(resume_from),
    )

    try:
        print(f"\n=== {EXP_NAME} === output: {output_dir}")
        print(f"Model: {MODEL_NAME}, subset={SUBSET}, n_cpus=50, max_steps=100")
        run_with_ray(exp, n_cpus=50)
    finally:
        print("\n--- post-run cleanup ---")
        leftover = INFRA.cleanup_orphaned_resources()
        if leftover and any(leftover.values()):
            n = sum(len(v) for v in leftover.values())
            print(f"Cleaned up {n} leftover resources from this run")


if __name__ == "__main__":
    main()
