import asyncio
import logging
from io import BytesIO

from cube.core import Action, Content, Observation, StepError
from cube_browser_tool import PlaywrightConfig
from PIL import Image
from playwright.async_api import Page as AsyncPage
from playwright.async_api import async_playwright

from cube_harness.action_spaces.browser_action_space import BrowserActionSpace
from cube_harness.tool import AsyncToolWithTelemetry
from cube_harness.utils import prune_html

logger = logging.getLogger(__name__)



def flatten_axtree(axtree_dict: dict | None) -> str:
    """
    Traverses accessibility tree dictionary and returns its markdown view.

    Args:
        axtree_dict: Accessibility tree from playwright page.accessibility.snapshot()
                     Structure: dict with 'role', 'name', 'value', 'children' keys

    Returns:
        String representation of the accessibility tree in markdown format
    """
    if axtree_dict is None:
        return ""

    def traverse_node(node: dict, depth: int = 0) -> list[str]:
        """Recursively traverse the accessibility tree and build markdown lines."""
        lines = []
        indent = "  " * depth  # 2 spaces per indent level

        # Extract node information
        role = node.get("role", "")
        name = node.get("name", "")
        value = node.get("value", "")

        # Build the node representation
        parts = []
        if role:
            parts.append(f"{role}:")
        if name.strip():
            parts.append(f"{name}")
        if value:
            parts.append(f"[value: {value}]")

        # Only add line if there's meaningful content
        if parts:
            line = f"{indent}{' '.join(parts)}"
            lines.append(line)

        # Recursively process children
        children = node.get("children", [])
        for child in children:
            child_lines = traverse_node(child, depth + 1)
            lines.extend(child_lines)

        return lines

    # Start traversal from root
    all_lines = traverse_node(axtree_dict, depth=0)
    return "\n".join(all_lines)
