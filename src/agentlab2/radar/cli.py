"""CLI entry point for agentlab radar."""

import argparse
import sys
from pathlib import Path

DEFAULT_RESULTS_DIR = "~/agentlab_results/al2"


def main() -> None:
    """Main entry point for the agentlab CLI."""
    parser = argparse.ArgumentParser(
        prog="agentlab",
        description="AgentLab2 - Agent evaluation and monitoring tools",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # radar subcommand
    radar_parser = subparsers.add_parser(
        "radar",
        help="Launch the experiment monitoring dashboard",
        description="Real-time monitoring and visualization of agent experiments.",
    )
    radar_parser.add_argument(
        "directory",
        type=str,
        nargs="?",
        default=DEFAULT_RESULTS_DIR,
        help=f"Path to experiment output directory (default: {DEFAULT_RESULTS_DIR})",
    )
    radar_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port number (default: auto-select available port)",
    )
    radar_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with hot reloading",
    )
    radar_parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio share link for remote access",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "radar":
        run_radar(args)


def run_radar(args: argparse.Namespace) -> None:
    """Run the radar dashboard."""
    from agentlab2.viewer import run_viewer

    results_dir = Path(args.directory).expanduser().resolve()

    if not results_dir.exists():
        print(f"Error: Directory does not exist: {results_dir}", file=sys.stderr)
        print("\nUsage: agentlab radar <directory>", file=sys.stderr)
        print("Example: agentlab radar ./output/my_experiment/", file=sys.stderr)
        sys.exit(1)

    print(f"Starting Radar dashboard for: {results_dir}")
    run_viewer(
        results_dir=results_dir,
        debug=args.debug,
        port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
