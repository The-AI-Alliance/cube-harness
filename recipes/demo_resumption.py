"""Demonstrate experiment resumption with MiniWob tasks.

This script:
1. Sets up an experiment with 10 MiniWob tasks
2. Runs the first few tasks (simulating an interruption)
3. Resumes the experiment to complete remaining tasks
"""

import logging
import sys
import time
from pathlib import Path

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.benchmarks.miniwob.benchmark import MiniWobBenchmark
from agentlab2.benchmarks.miniwob.task import MiniWobTask
from agentlab2.experiment import Experiment
from agentlab2.llm import LLMConfig
from agentlab2.tools.playwright import PlaywrightConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(name)s:%(lineno)d %(funcName)s() - %(message)s",
)
logger = logging.getLogger(__name__)


class LimitedMiniWobBenchmark(MiniWobBenchmark):
    """MiniWob benchmark limited to a specific number of tasks."""

    max_tasks: int = 10

    def load_tasks(self) -> list[MiniWobTask]:
        """Load tasks but limit to max_tasks."""
        all_tasks = super().load_tasks()
        limited_tasks = all_tasks[: self.max_tasks]
        logger.info(f"Limited benchmark to {len(limited_tasks)} tasks (from {len(all_tasks)} total)")
        return limited_tasks


def run_initial_evaluation(exp: Experiment, interrupt_after: int = 3) -> None:
    """
    Run initial evaluation, simulating an interruption after a few tasks.

    Args:
        exp: The experiment to run
        interrupt_after: Number of tasks to complete before "interruption"
    """
    logger.info("=" * 80)
    logger.info("PHASE 1: Initial Evaluation (will be interrupted)")
    logger.info("=" * 80)

    exp.benchmark.setup()
    try:
        episodes = exp.get_episodes_to_run()
        logger.info(f"Created {len(episodes)} episodes")

        # Run only the first few episodes to simulate interruption
        logger.info(f"Running first {interrupt_after} episodes (simulating interruption after)...")
        trajectories = []
        failures = {}

        for i, episode in enumerate(episodes[:interrupt_after]):
            logger.info(f"\n--- Running Episode {i + 1}/{interrupt_after} (Task: {episode.config.task_id}) ---")
            try:
                trajectory = episode.run()
                trajectories.append(trajectory)
                logger.info(f"✓ Episode {i + 1} completed successfully")
            except Exception as e:
                logger.exception(f"✗ Episode {i + 1} failed: {e}")
                failures[episode.config.task_id] = str(e)

        logger.info(f"\n✓ Completed {len(trajectories)} episodes successfully")
        logger.info(f"✗ {len(failures)} episodes failed")
        logger.info("\n⚠ Simulating interruption (e.g., Ctrl+C, system crash, etc.)")
        logger.info("   Episode configs have been saved and can be resumed later")

    finally:
        exp.benchmark.close()


def resume_evaluation(exp: Experiment) -> None:
    """
    Resume evaluation by running unstarted and failed episodes.

    Args:
        exp: The experiment to resume
    """
    from agentlab2.exp_runner import run_sequentially

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: Resuming Evaluation")
    logger.info("=" * 80)

    # First, resume unstarted episodes (those that were never run)
    logger.info("\n--- Resuming unstarted episodes ---")
    exp.resume = True
    exp.retry_failed = False
    unstarted_results = run_sequentially(exp)

    if unstarted_results.tasks_num > 0:
        logger.info(f"✓ Resumed {unstarted_results.tasks_num} unstarted episodes")
        logger.info(f"  - Successful: {len(unstarted_results.trajectories)}")
        logger.info(f"  - Failed: {len(unstarted_results.failures)}")
    else:
        logger.info("✓ No unstarted episodes to resume")

    # Then, resume failed episodes (those that started but failed)
    logger.info("\n--- Resuming failed episodes ---")
    exp.resume = False
    exp.retry_failed = True
    failed_results = run_sequentially(exp)

    if failed_results.tasks_num > 0:
        logger.info(f"✓ Resumed {failed_results.tasks_num} failed episodes")
        logger.info(f"  - Successful: {len(failed_results.trajectories)}")
        logger.info(f"  - Failed: {len(failed_results.failures)}")
    else:
        logger.info("✓ No failed episodes to resume")

    # Print combined statistics
    total_trajectories = len(unstarted_results.trajectories) + len(failed_results.trajectories)
    total_failures = len(unstarted_results.failures) + len(failed_results.failures)

    logger.info("\n" + "=" * 80)
    logger.info("RESUMPTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total trajectories completed: {total_trajectories}")
    logger.info(f"Total failures: {total_failures}")
    logger.info(f"Experiment output directory: {exp.output_dir}")


def main(interrupt_after: int = 3, resume: bool = True) -> None:
    """
    Main function to demonstrate experiment resumption.

    Args:
        interrupt_after: Number of tasks to complete before interruption
        resume: Whether to resume after interruption
    """
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path.home() / "agentlab_results" / "al2" / f"resumption_demo_{current_datetime}"

    logger.info("=" * 80)
    logger.info("EXPERIMENT RESUMPTION DEMONSTRATION")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Will interrupt after {interrupt_after} tasks")
    logger.info(f"Will resume: {resume}")

    # Setup experiment with limited benchmark
    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)
    agent_config = ReactAgentConfig(llm_config=llm_config)
    tool_config = PlaywrightConfig(use_screenshot=True, headless=True)
    benchmark = LimitedMiniWobBenchmark(tool_config=tool_config, max_tasks=10)

    exp = Experiment(
        name="resumption_demo",
        output_dir=output_dir,
        agent_config=agent_config,
        benchmark=benchmark,
    )

    # Save experiment config
    exp.save_config()

    # Phase 1: Run initial evaluation (will be interrupted)
    run_initial_evaluation(exp, interrupt_after=interrupt_after)

    # Phase 2: Resume evaluation
    if resume:
        resume_evaluation(exp)
    else:
        logger.info("\n⚠ Resumption skipped (resume=False)")
        logger.info(f"   To resume later, load the experiment from: {output_dir / 'experiment_config.json'}")
        logger.info("   Then set exp.resume=True and/or exp.retry_failed=True, then call run_sequentially(exp)")

    logger.info("\n" + "=" * 80)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    # Parse command line arguments
    interrupt_after = 3
    resume = True

    if len(sys.argv) > 1:
        try:
            interrupt_after = int(sys.argv[1])
        except ValueError:
            logger.warning(f"Invalid interrupt_after value: {sys.argv[1]}, using default: 3")

    if len(sys.argv) > 2:
        resume = sys.argv[2].lower() in ("true", "1", "yes", "y")

    main(interrupt_after=interrupt_after, resume=resume)
