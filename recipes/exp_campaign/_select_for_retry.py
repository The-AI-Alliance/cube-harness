"""Inspect a result dir and report what `Experiment(resume=True)` will do per task.

Auto-resume only re-queues tasks whose `status.json` is in `RETRIABLE_STATUSES =
{FAILED, CANCELLED, STALE}` (with `RUNNING` sweep_stale_statuses → STALE first).
QUEUED tasks from killed runs get skipped, and content-policy / malformed-json
FAILED tasks get retried even though they're real eval signal.

Modes:
    Default — classify by status.json only. Reports what auto-resume will do.
    --retriage — additionally flag COMPLETED-reward=0 tasks where
        `had_step_errors=True` as RETRY_VIA_RETRIAGE: those tasks hit a StepError
        mid-run that the *old* code handled by terminating the episode. With
        fix/computer-tool-stepError-on-execute-action merged, that StepError
        becomes recoverable feedback (error message + fresh screenshot), so the
        episode continues and the agent can correct course. Worth retrying.

Usage:
    python _select_for_retry.py <result_dir>                    # dry-run, default
    python _select_for_retry.py <result_dir> --retriage          # dry-run, deep
    python _select_for_retry.py <result_dir> --retriage --apply
            [--force-running]   # also force RUNNING → FAILED (skip sweep wait)
            [--skip-wasteful]   # mark content_policy / json errors → MAX_STEPS_REACHED
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cube_harness.episode_status import EpisodeStatus  # noqa: E402

WASTEFUL_AGENT_ERROR_TYPES = {"ContentPolicyViolationError", "JSONDecodeError"}


def classify(st: EpisodeStatus | None, retriage: bool = False) -> str:
    """Plan label.

    With retriage=False: status-driven only. With retriage=True: additionally flag
    COMPLETED-reward=0 tasks where `had_step_errors=True` as RETRY_VIA_RETRIAGE,
    since those terminated mid-run on a StepError that's now recoverable.
    """
    if st is None:
        return "AUTO_FRESH"

    s = st.status

    if s == "COMPLETED":
        if (st.reward or 0) > 0:
            return "KEEP_PASS"
        if retriage and st.had_step_errors:
            return "RETRY_VIA_RETRIAGE"
        return "KEEP_REAL_FAIL"
    if s == "MAX_STEPS_REACHED":
        return "KEEP_AGENT_MAX_STEPS"
    if s == "FAILED":
        if st.error_type in WASTEFUL_AGENT_ERROR_TYPES:
            return "WASTE_RETRY"
        return "AUTO_RETRY"
    if s in ("CANCELLED", "STALE"):
        return "AUTO_RETRY"
    if s == "QUEUED":
        return "NEEDS_FORCE_QUEUED"
    if s == "RUNNING":
        return "NEEDS_FORCE_RUNNING"
    return f"UNKNOWN_{s}"


def force_to_failed(st: EpisodeStatus, path: Path, reason: str = "") -> None:
    st.status = "FAILED"
    st.retry_count = 0
    st.ended_at = st.ended_at or time.time()
    st.error_type = st.error_type or "ForcedRetry"
    suffix = f" [forced FAILED via _select_for_retry: {reason}]" if reason else " [forced FAILED via _select_for_retry]"
    st.error_message = (st.error_message or "") + suffix
    path.write_text(json.dumps(asdict(st), indent=2))


def force_to_max_steps(st: EpisodeStatus, path: Path) -> None:
    st.status = "MAX_STEPS_REACHED"
    st.error_message = (st.error_message or "") + " [skipped via _select_for_retry --skip-wasteful]"
    path.write_text(json.dumps(asdict(st), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--retriage", action="store_true", help="re-classify tasks by reading episode.log")
    parser.add_argument("--apply", action="store_true", help="actually rewrite status.json files")
    parser.add_argument("--force-running", action="store_true", help="also force RUNNING → FAILED")
    parser.add_argument("--skip-wasteful", action="store_true", help="content_policy / json errors → MAX_STEPS_REACHED")
    args = parser.parse_args()

    if not args.result_dir.is_dir():
        sys.exit(f"not a directory: {args.result_dir}")
    episodes_root = args.result_dir / "episodes"
    if not episodes_root.is_dir():
        sys.exit(f"no episodes/ subdir in {args.result_dir}")

    live_dirs = sorted(d for d in episodes_root.iterdir() if d.is_dir() and ".archived_" not in d.name)

    plans: Counter[str] = Counter()
    actions: list[tuple[str, Path, EpisodeStatus, str]] = []  # (kind, path, status, summary)

    for ep_dir in live_dirs:
        status_path = ep_dir / "status.json"
        st = EpisodeStatus.read(status_path)
        plan = classify(st, retriage=args.retriage)
        plans[plan] += 1

        if plan == "NEEDS_FORCE_QUEUED" and st is not None:
            actions.append(("force_failed", status_path, st, f"QUEUED → FAILED ({ep_dir.name[:50]})"))
        elif plan == "NEEDS_FORCE_RUNNING" and args.force_running and st is not None:
            actions.append(("force_failed", status_path, st, f"RUNNING → FAILED ({ep_dir.name[:50]})"))
        elif plan == "RETRY_VIA_RETRIAGE" and st is not None:
            actions.append(
                ("force_failed_retriage", status_path, st, f"{st.status} → FAILED via retriage ({ep_dir.name[:50]})")
            )
        elif plan == "WASTE_RETRY" and args.skip_wasteful and st is not None:
            actions.append(
                ("skip_wasteful", status_path, st, f"{st.error_type} → MAX_STEPS_REACHED ({ep_dir.name[:50]})")
            )

    print(f"\n=== {args.result_dir.name} ===")
    print(f"Live episode dirs: {len(live_dirs)}{'  (retriage on)' if args.retriage else ''}")
    label_order = [
        "KEEP_PASS",
        "KEEP_REAL_FAIL",
        "KEEP_AGENT_MAX_STEPS",
        "AUTO_RETRY",
        "RETRY_VIA_RETRIAGE",
        "WASTE_RETRY",
        "NEEDS_FORCE_QUEUED",
        "NEEDS_FORCE_RUNNING",
        "AUTO_FRESH",
    ]
    for label in label_order:
        if plans.get(label):
            print(f"  {label:<22} {plans[label]:>4}")
    other = {k: v for k, v in plans.items() if k not in label_order}
    for k, v in other.items():
        print(f"  {k:<22} {v:>4}  (unclassified)")

    print("\nActions queued (would be applied with --apply):")
    if not actions:
        print("  (none)")
    else:
        action_kinds = Counter(k for k, _, _, _ in actions)
        for k, c in action_kinds.items():
            print(f"  {k:<22} x{c}")

    if not args.apply:
        print("\nDry-run: pass --apply to make these changes.")
        return

    for kind, path, st, _summary in actions:
        if kind in ("force_failed", "force_failed_retriage"):
            force_to_failed(st, path, reason="retriage" if kind == "force_failed_retriage" else "")
        elif kind == "skip_wasteful":
            force_to_max_steps(st, path)
    print(f"\nApplied {len(actions)} status.json rewrite(s).")


if __name__ == "__main__":
    main()
