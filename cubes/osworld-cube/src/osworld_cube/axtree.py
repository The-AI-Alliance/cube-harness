"""Accessibility tree processing utilities for OSWorld.

Ported from kusha/AgentLab2/src/agentlab2/benchmarks/osworld/osworld_axtree.py.

Provides:
    linearize_accessibility_tree(xml_str, platform) -> str
        Convert XML accessibility tree to a tab-separated table for the agent.

    tag_screenshot(screenshot_bytes, xml_str, platform) -> (marks, drew_nodes, tagged_bytes, element_list)
        Draw numbered bounding boxes on a screenshot (Set-of-Marks).
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

# XML namespace URLs for accessibility tree attributes
attributes_ns_ubuntu = "https://accessibility.windows.example.org/ns/attributes"
attributes_ns_windows = "https://accessibility.windows.example.org/ns/attributes"
state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
state_ns_windows = "https://accessibility.windows.example.org/ns/state"
component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
component_ns_windows = "https://accessibility.windows.example.org/ns/component"
value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
value_ns_windows = "https://accessibility.windows.example.org/ns/value"
class_ns_windows = "https://accessibility.windows.example.org/ns/class"


def _get_ns(platform: str) -> tuple[str, str, str, str]:
    """Return (attributes_ns, state_ns, component_ns, value_ns) for the given platform."""
    if platform == "ubuntu":
        return attributes_ns_ubuntu, state_ns_ubuntu, component_ns_ubuntu, value_ns_ubuntu
    if platform == "windows":
        return attributes_ns_windows, state_ns_windows, component_ns_windows, value_ns_windows
    raise ValueError(f"Invalid platform '{platform}': must be 'ubuntu' or 'windows'")


def judge_node(node: ET.Element, platform: str = "ubuntu", check_image: bool = False) -> bool:
    """Return True if this accessibility tree node should be included in the output.

    Filters to visible, enabled, and interactable nodes that have a name or text.
    """
    _, _state_ns, _component_ns, _ = _get_ns(platform)

    keeps: bool = (
        node.tag.startswith("document")
        or node.tag.endswith("item")
        or node.tag.endswith("button")
        or node.tag.endswith("heading")
        or node.tag.endswith("label")
        or node.tag.endswith("scrollbar")
        or node.tag.endswith("searchbox")
        or node.tag.endswith("textbox")
        or node.tag.endswith("link")
        or node.tag.endswith("tabelement")
        or node.tag.endswith("textfield")
        or node.tag.endswith("textarea")
        or node.tag.endswith("menu")
        or node.tag
        in {
            "alert",
            "canvas",
            "check-box",
            "combo-box",
            "entry",
            "icon",
            "image",
            "paragraph",
            "scroll-bar",
            "section",
            "slider",
            "static",
            "table-cell",
            "terminal",
            "text",
            "netuiribbontab",
            "start",
            "trayclockwclass",
            "traydummysearchcontrol",
            "uiimage",
            "uiproperty",
            "uiribboncommandbar",
        }
    )

    keeps = (
        keeps
        and (
            platform == "ubuntu"
            and node.get(f"{{{_state_ns}}}showing", "false") == "true"
            and node.get(f"{{{_state_ns}}}visible", "false") == "true"
            or platform == "windows"
            and node.get(f"{{{_state_ns}}}visible", "false") == "true"
        )
        and (
            node.get(f"{{{_state_ns}}}enabled", "false") == "true"
            or node.get(f"{{{_state_ns}}}editable", "false") == "true"
            or node.get(f"{{{_state_ns}}}expandable", "false") == "true"
            or node.get(f"{{{_state_ns}}}checkable", "false") == "true"
        )
        and (
            node.get("name", "") != ""
            or node.text is not None
            and len(node.text) > 0
            or check_image
            and node.get("image", "false") == "true"
        )
    )

    coords: Tuple[int, int] = eval(node.get(f"{{{_component_ns}}}screencoord", "(-1, -1)"))
    sizes: Tuple[int, int] = eval(node.get(f"{{{_component_ns}}}size", "(-1, -1)"))
    keeps = keeps and coords[0] >= 0 and coords[1] >= 0 and sizes[0] > 0 and sizes[1] > 0
    return keeps


def filter_nodes(root: ET.Element, platform: str = "ubuntu", check_image: bool = False) -> list[ET.Element]:
    """Return all visible and interactable nodes from the accessibility tree."""
    return [node for node in root.iter() if judge_node(node, platform, check_image)]


def draw_bounding_boxes(
    nodes: list[ET.Element],
    image_file_content: bytes,
    down_sampling_ratio: float = 1.0,
    platform: str = "ubuntu",
) -> Tuple[list, list, str, bytes]:
    """Draw numbered bounding boxes on a screenshot for the given accessibility nodes.

    Returns:
        marks:             list of [x, y, w, h] bounding boxes (original coords)
        drew_nodes:        list of ET.Element nodes that were actually drawn
        text_informations: tab-separated table of node info (index/tag/name/text)
        image_content:     annotated screenshot as PNG bytes
    """
    _, _state_ns, _component_ns, _value_ns = _get_ns(platform)

    image = Image.open(io.BytesIO(image_file_content))
    if float(down_sampling_ratio) != 1.0:
        image = image.resize(
            (
                int(image.size[0] * down_sampling_ratio),
                int(image.size[1] * down_sampling_ratio),
            )
        )
    draw = ImageDraw.Draw(image)
    marks: list = []
    drew_nodes: list = []
    text_informations: List[str] = ["index\ttag\tname\ttext"]

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    index = 1
    for _node in nodes:
        coords_str = _node.attrib.get(f"{{{_component_ns}}}screencoord")
        size_str = _node.attrib.get(f"{{{_component_ns}}}size")
        if not coords_str or not size_str:
            continue
        try:
            coords = tuple(map(int, coords_str.strip("()").split(", ")))
            size = tuple(map(int, size_str.strip("()").split(", ")))
            original_coords = coords
            original_size = size

            if float(down_sampling_ratio) != 1.0:
                coords = tuple(int(c * down_sampling_ratio) for c in coords)
                size = tuple(int(s * down_sampling_ratio) for s in size)

            if size[0] <= 0 or size[1] <= 0:
                raise ValueError(f"Size must be positive, got: {size}")

            bottom_right = (coords[0] + size[0], coords[1] + size[1])
            if bottom_right[0] < coords[0] or bottom_right[1] < coords[1]:
                raise ValueError(f"Invalid coordinates: coords={coords}, size={size}")

            # Skip single-colour (blank) regions
            cropped = image.crop((*coords, *bottom_right))
            if len(set(list(cropped.getdata()))) == 1:
                continue

            draw.rectangle([coords, bottom_right], outline="red", width=1)
            text_pos = (coords[0], bottom_right[1])
            text_bbox: Tuple[int, int, int, int] = draw.textbbox(text_pos, str(index), font=font, anchor="lb")
            draw.rectangle(text_bbox, fill="black")
            draw.text(text_pos, str(index), font=font, anchor="lb", fill="white")

            marks.append([original_coords[0], original_coords[1], original_size[0], original_size[1]])
            drew_nodes.append(_node)

            # Build node text for the element table
            if _node.text:
                node_text = _node.text if '"' not in _node.text else '"{:}"'.format(_node.text.replace('"', '""'))
            elif _node.get(f"{{{class_ns_windows}}}class", "").endswith("EditWrapper") and _node.get(
                f"{{{_value_ns}}}value"
            ):
                raw = _node.get(f"{{{_value_ns}}}value", "")
                node_text = raw if '"' not in raw else '"{:}"'.format(raw.replace('"', '""'))
            else:
                node_text = '""'

            text_informations.append(f"{index}\t{_node.tag}\t{_node.get('name', '')}\t{node_text}")
            index += 1

        except (ValueError, SyntaxError):
            pass

    out = io.BytesIO()
    image.save(out, format="PNG")
    return marks, drew_nodes, "\n".join(text_informations), out.getvalue()


def linearize_accessibility_tree(accessibility_tree: str, platform: str = "ubuntu") -> str:
    """Convert an XML accessibility tree to a tab-separated table for the agent.

    Columns: tag, name, text, class, description, position (top-left x&y), size (w&h)

    Args:
        accessibility_tree: Raw XML string from desktop_env
        platform: "ubuntu" or "windows"

    Returns:
        Tab-separated table as a single string.
    """
    _attributes_ns, _state_ns, _component_ns, _value_ns = _get_ns(platform)

    filtered_nodes = filter_nodes(ET.fromstring(accessibility_tree), platform)
    rows = ["tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"]

    for node in filtered_nodes:
        if node.text:
            text = node.text if '"' not in node.text else '"{:}"'.format(node.text.replace('"', '""'))
        elif node.get(f"{{{class_ns_windows}}}class", "").endswith("EditWrapper") and node.get(f"{{{_value_ns}}}value"):
            raw = node.get(f"{{{_value_ns}}}value", "")
            text = raw if '"' not in raw else '"{:}"'.format(raw.replace('"', '""'))
        else:
            text = '""'

        cls = (
            node.get(f"{{{_attributes_ns}}}class", "")
            if platform == "ubuntu"
            else node.get(f"{{{class_ns_windows}}}class", "")
        )
        rows.append(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                node.tag,
                node.get("name", ""),
                text,
                cls,
                node.get(f"{{{_attributes_ns}}}description", ""),
                node.get(f"{{{_component_ns}}}screencoord", ""),
                node.get(f"{{{_component_ns}}}size", ""),
            )
        )

    return "\n".join(rows)


def tag_screenshot(
    screenshot: bytes, accessibility_tree: str, platform: str = "ubuntu"
) -> Tuple[list, list, bytes, str]:
    """Annotate a screenshot with numbered bounding boxes for interactive elements.

    Args:
        screenshot: PNG screenshot bytes
        accessibility_tree: XML string from desktop_env
        platform: "ubuntu" or "windows"

    Returns:
        marks:         list of [x, y, w, h] for each drawn element
        drew_nodes:    ET.Element nodes that were drawn
        tagged_screenshot: annotated PNG bytes
        element_list:  tab-separated element table (index/tag/name/text)
    """
    nodes = filter_nodes(ET.fromstring(accessibility_tree), platform=platform, check_image=True)
    marks, drew_nodes, element_list, tagged_screenshot = draw_bounding_boxes(nodes, screenshot, platform=platform)
    return marks, drew_nodes, tagged_screenshot, element_list
