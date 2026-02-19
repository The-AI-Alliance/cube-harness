"""Interactive debug shell for OSWorld VM.

Boots a VM to its initial state and enters an interactive loop where you type
actions and see the resulting screenshot after each one.

Prerequisites:
    pip install desktop-env  (or uv add desktop-env)

Usage:
    uv run recipes/debug_osworld.py
    uv run recipes/debug_osworld.py --provider vmware --no-headless

Available commands:
    screenshot              - Capture and display current screen
    click X Y               - Left click at (X, Y)
    right_click X Y         - Right click at (X, Y)
    double_click X Y        - Double click at (X, Y)
    move_to X Y             - Move cursor to (X, Y)
    drag_to X Y             - Drag cursor to (X, Y)
    scroll DX DY            - Scroll by (DX, DY)
    typing TEXT             - Type a text string
    press KEY               - Press and release a key (e.g. enter, space, ctrl)
    key_down KEY            - Press a key without releasing
    key_up KEY              - Release a held key
    hotkey KEY1 KEY2 ...    - Press key combination (e.g. hotkey ctrl c)
    mouse_down [BUTTON]     - Press mouse button (default: left)
    mouse_up [BUTTON]       - Release mouse button (default: left)
    wait                    - Send a wait/no-op action
    help                    - Show this help
    quit / exit / q         - Close VM and exit

Screenshots are saved to /tmp/osworld_debug/ and opened in your default viewer.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from agentlab2.tools.computer import Computer, ComputerConfig

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

SCREENSHOT_DIR = Path("/tmp/osworld_debug")

# Minimal task config used to boot the VM to its initial snapshot state.
# No setup scripts or evaluator — just boot and interact.
MINIMAL_TASK_CONFIG: dict = {
    "id": "debug",
    "instruction": "Debug session — no specific task",
    "config": [],
    "evaluator": {"func": "infeasible", "expected": {}},
}

HELP_TEXT = """
Available commands:
  screenshot              Capture and display current screen
  click X Y               Left click at (X, Y)
  right_click X Y         Right click at (X, Y)
  double_click X Y        Double click at (X, Y)
  move_to X Y             Move cursor to (X, Y)
  drag_to X Y             Drag cursor to (X, Y)
  scroll DX DY            Scroll by (DX, DY)  [positive DY = up]
  typing TEXT             Type a text string
  press KEY               Press and release a key (enter, space, ctrl, ...)
  key_down KEY            Press a key without releasing
  key_up KEY              Release a held key
  hotkey KEY1 KEY2 ...    Press a key combo  (e.g. hotkey ctrl c)
  mouse_down [BUTTON]     Press mouse button  (left/right/middle, default left)
  mouse_up [BUTTON]       Release mouse button
  wait                    Send a wait/no-op action
  help                    Show this message
  quit / exit / q         Close VM and exit
"""


def _save_and_show_screenshot(computer: Computer) -> None:
    """Capture current VM screenshot, save it, and open in default viewer."""
    obs = computer.get_observation()
    for content in obs.contents:
        if content.name == "screenshot":
            img = content.data  # PIL Image
            SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%H%M%S_%f")
            path = SCREENSHOT_DIR / f"screenshot_{timestamp}.png"
            img.save(path)
            print(f"  Screenshot: {path}")
            img.show()
            return
    print("  (no screenshot available)")


def _execute_command(line: str, computer: Computer) -> bool:
    """Parse and execute one command line.

    Returns False if the user requested to quit, True otherwise.
    """
    parts = line.strip().split()
    if not parts:
        return True

    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in ("quit", "exit", "q"):
        return False

    if cmd == "help":
        print(HELP_TEXT)
        return True

    if cmd == "screenshot":
        _save_and_show_screenshot(computer)
        return True

    # Dispatch to Computer methods
    try:
        result: str
        if cmd == "click":
            x, y = int(args[0]), int(args[1])
            result = computer.click(x=x, y=y)
        elif cmd == "right_click":
            x, y = int(args[0]), int(args[1])
            result = computer.right_click(x=x, y=y)
        elif cmd == "double_click":
            x, y = int(args[0]), int(args[1])
            result = computer.double_click(x=x, y=y)
        elif cmd == "move_to":
            x, y = int(args[0]), int(args[1])
            result = computer.move_to(x=x, y=y)
        elif cmd == "drag_to":
            x, y = int(args[0]), int(args[1])
            result = computer.drag_to(x=x, y=y)
        elif cmd == "scroll":
            dx, dy = int(args[0]), int(args[1])
            result = computer.scroll(dx=dx, dy=dy)
        elif cmd == "typing":
            text = " ".join(args)
            result = computer.typing(text)
        elif cmd == "press":
            result = computer.press(args[0])
        elif cmd == "key_down":
            result = computer.key_down(args[0])
        elif cmd == "key_up":
            result = computer.key_up(args[0])
        elif cmd == "hotkey":
            result = computer.hotkey(list(args))
        elif cmd == "mouse_down":
            button = args[0] if args else "left"
            result = computer.mouse_down(button=button)
        elif cmd == "mouse_up":
            button = args[0] if args else "left"
            result = computer.mouse_up(button=button)
        elif cmd == "wait":
            result = computer.wait()
        else:
            print(f"  Unknown command: '{cmd}'. Type 'help' for available commands.")
            return True

        print(f"  Result: {result}")

    except (ValueError, IndexError):
        print(f"  Bad arguments for '{cmd}'. Type 'help' for usage.")
        return True
    except Exception as exc:
        print(f"  Action error: {exc}")
        logger.exception("Action execution failed")
        return True

    # Auto-capture screenshot to show impact of the action
    _save_and_show_screenshot(computer)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive OSWorld debug shell — boot a VM and send actions interactively.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_TEXT,
    )
    parser.add_argument("--provider", default="docker", help="VM provider (docker, vmware, virtualbox)")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless (default)")
    parser.add_argument("--no-headless", dest="headless", action="store_false", help="Show VM GUI (VMware only)")
    parser.add_argument("--require-axtree", action="store_true", default=False, help="Also fetch accessibility tree")
    args = parser.parse_args()

    print("=" * 60)
    print("OSWorld Debug Shell")
    print("=" * 60)
    print(f"Provider : {args.provider}")
    print(f"Headless : {args.headless}")
    print(f"A11y tree: {args.require_axtree}")
    print("Booting VM — this takes ~60 seconds...")
    print("=" * 60)

    config = ComputerConfig(
        provider=args.provider,
        headless=args.headless,
        require_a11y_tree=args.require_axtree,
    )

    computer = config.make()

    try:
        computer.setup_task(MINIMAL_TASK_CONFIG)

        print("\nVM ready. Taking initial screenshot...")
        _save_and_show_screenshot(computer)
        print("\nType 'help' for available commands. Type 'quit' to exit.\n")

        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nInterrupted — exiting.")
                break

            if not _execute_command(line, computer):
                break

    finally:
        print("Closing VM...")
        computer.close()
        print("Done.")


if __name__ == "__main__":
    main()
