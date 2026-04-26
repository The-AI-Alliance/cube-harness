"""Fast cheat-based smoke test for WorkArena create tasks.

Two test modes:
  cheat        — task.cheat() + validate() via classic #sysverb_insert button
  react-submit — setup, click visible Submit (without filling), log raw POST body verb

Usage:
    cd cube-harness
    uv run meta_agent/test_workarena_cheat.py
    uv run meta_agent/test_workarena_cheat.py create-hardware-asset
    uv run meta_agent/test_workarena_cheat.py create-change-request react-submit
"""

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

from playwright.sync_api import sync_playwright  # noqa: E402

from browsergym.workarena.instance import SNowInstance  # noqa: E402
from browsergym.workarena.tasks.form import (  # noqa: E402
    CreateChangeRequestTask,
    CreateHardwareAssetTask,
)

TASKS = {
    "create-change-request": CreateChangeRequestTask,
    "create-hardware-asset": CreateHardwareAssetTask,
}

SEED = 700


def run_cheat_test(name: str, task_cls, seed: int) -> bool:
    """Classic path: cheat() uses #sysverb_insert (always sysverb_insert)."""
    instance = SNowInstance()
    print(f"\n{'='*60}")
    print(f"[cheat] {name}  seed={seed}", flush=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        task = task_cls(seed=seed, instance=instance)
        t0 = time.time()
        task.setup(page=page)
        print(f"  setup: {time.time()-t0:.1f}s", flush=True)
        task.cheat(page=page, chat_messages=[])
        reward, done, _, ri = task.validate(page, [])
        print(f"  reward={reward} done={done} msg={ri.get('message','')[:120]}", flush=True)
        task.teardown()
        browser.close()

    passed = reward == 1.0 and done is True
    print(f"  RESULT: {'PASS ✓' if passed else 'FAIL ✗'}", flush=True)
    return passed


def run_react_submit_test(name: str, task_cls, seed: int) -> bool:
    """Diagnostic: what verb does the visible Submit button send?

    Clicks the React Submit button (no field filling) and prints the raw POST body.
    This tells us:
      - If it sends sysverb_insert_and_stay → Patch 3 is needed
      - If it sends sysverb_insert → button already uses correct verb
    Does NOT check reward (no fields filled, record will be invalid).
    """
    instance = SNowInstance()
    print(f"\n{'='*60}")
    print(f"[react-submit] {name}  seed={seed}", flush=True)

    captured_bodies: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        # Fresh page — do NOT reuse from cheat test
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.set_default_navigation_timeout(90000)

        # Log ALL POST bodies BEFORE setup so we see what goes out before Patch 3
        def _log_all_posts(route, request):
            if request.method == "POST":
                body = request.post_data or ""
                url = request.url
                captured_bodies.append(f"URL={url[-80:]}\nBODY={body[:200]}")
            route.continue_()

        page.route("**/*", _log_all_posts)

        task = task_cls(seed=seed, instance=instance)
        t0 = time.time()
        task.setup(page=page)  # Patch 3 registered here (LIFO: fires before _log_all_posts)
        print(f"  setup: {time.time()-t0:.1f}s", flush=True)
        print(f"  js_api: {task.form_js_selector}", flush=True)

        # Click visible Submit button — try several selectors
        t1 = time.time()
        clicked = False
        for selector in [
            "button:has-text('Submit')",
            "[aria-label='Submit']",
            ".btn-primary",
        ]:
            try:
                btn = page.locator(selector).first
                if btn.count() > 0:
                    btn.click(timeout=3000)
                    clicked = True
                    print(f"  clicked outer: {selector!r}", flush=True)
                    break
            except Exception:
                pass

        if not clicked:
            iframe_loc = page.frame_locator("#gsft_main")
            for selector in ["button.btn-primary:visible", "button:has-text('Submit'):visible"]:
                try:
                    btn = iframe_loc.locator(selector).first
                    btn.click(timeout=3000)
                    clicked = True
                    print(f"  clicked iframe: {selector!r}", flush=True)
                    break
                except Exception:
                    pass

        if not clicked:
            print("  WARNING: no Submit button found", flush=True)

        print(f"  click took {time.time()-t1:.1f}s, waiting for POST...", flush=True)
        page.wait_for_timeout(8000)

        print(f"\n  === Captured POST bodies ({len(captured_bodies)}) ===", flush=True)
        for i, body in enumerate(captured_bodies):
            print(f"  [{i}] {body[:300]}", flush=True)

        # Summarize
        raw_verbs = []
        for body in captured_bodies:
            if "sysverb_insert_and_stay" in body:
                raw_verbs.append("sysverb_insert_and_stay")
            elif "sysverb_insert" in body:
                raw_verbs.append("sysverb_insert")

        print(f"\n  DIAGNOSTIC: raw verbs sent to server = {raw_verbs}", flush=True)
        if "sysverb_insert_and_stay" in raw_verbs:
            print("  → React Submit sends sysverb_insert_and_stay (Patch 3 normalized it)", flush=True)
        elif "sysverb_insert" in raw_verbs:
            print("  → React Submit sends sysverb_insert (correct, no normalization needed)", flush=True)
        else:
            print("  → No sysverb POST captured (Submit may use different mechanism)", flush=True)

        task.teardown()
        browser.close()

    # Diagnostic test always "passes" if it ran without crashing
    passed = len(captured_bodies) >= 0  # always true
    print(f"  RESULT: DIAGNOSTIC COMPLETE", flush=True)
    return True


if __name__ == "__main__":
    filter_name = sys.argv[1] if len(sys.argv) > 1 else None
    mode = sys.argv[2] if len(sys.argv) > 2 else "cheat"

    tasks_to_run = {k: v for k, v in TASKS.items() if not filter_name or filter_name in k}

    results: dict[str, bool] = {}

    for name, cls in tasks_to_run.items():
        if mode in ("cheat", "all"):
            key = f"{name}/cheat"
            try:
                results[key] = run_cheat_test(name, cls, SEED)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                results[key] = False

        if mode in ("react-submit", "all"):
            key2 = f"{name}/react-submit"
            try:
                results[key2] = run_react_submit_test(name, cls, SEED)
            except Exception as e:
                import traceback
                traceback.print_exc()
                results[key2] = False

    print(f"\n{'='*60}")
    print("Summary:")
    for key, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {key}", flush=True)

    sys.exit(0 if all(results.values()) else 1)
