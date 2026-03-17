"""Tests for cube_harness.agents.legacy_generic_agent prompt generation."""

import pytest
from cube.core import ActionSchema, Content, Observation
from PIL import Image

from cube_harness.agents.legacy_generic_agent import (
    BeCautious,
    Criticise,
    GenericAgent,
    GenericAgentConfig,
    GenericPromptFlags,
    Hints,
    History,
    HistoryStep,
    MainPrompt,
    Memory,
    ObsFlags,
    Plan,
    PromptElement,
    ShrinkableObservation,
    Think,
    Trunkater,
    parse_html_tags,
)
from cube_harness.llm import LLMConfig

# ============================================================================
# Test parse_html_tags
# ============================================================================


class TestParseHtmlTags:
    """Tests for the parse_html_tags function."""

    def test_parse_single_required_tag(self) -> None:
        """Test parsing a single required tag."""
        text = "<think>This is my thought</think>"
        result = parse_html_tags(text, keys=["think"])
        assert result == {"think": "This is my thought"}

    def test_parse_multiple_tags(self) -> None:
        """Test parsing multiple tags."""
        text = "<think>My thought</think><plan>My plan</plan>"
        result = parse_html_tags(text, keys=["think", "plan"])
        assert result["think"] == "My thought"
        assert result["plan"] == "My plan"

    def test_parse_optional_tags_present(self) -> None:
        """Test parsing optional tags when present."""
        text = "<think>Thought</think><memory>Memory content</memory>"
        result = parse_html_tags(text, keys=["think"], optional_keys=["memory"])
        assert result["think"] == "Thought"
        assert result["memory"] == "Memory content"

    def test_parse_optional_tags_missing(self) -> None:
        """Test parsing optional tags when missing."""
        text = "<think>Thought</think>"
        result = parse_html_tags(text, keys=["think"], optional_keys=["memory"])
        assert result["think"] == "Thought"
        assert "memory" not in result

    def test_parse_missing_required_tag_raises(self) -> None:
        """Test that missing required tag raises ValueError."""
        text = "<think>Thought</think>"
        with pytest.raises(ValueError, match="Required tag <plan> not found"):
            parse_html_tags(text, keys=["think", "plan"])

    def test_parse_multiline_content(self) -> None:
        """Test parsing tag with multiline content."""
        text = """<plan>
Step 1: Do this
Step 2: Do that
Step 3: Finish
</plan>"""
        result = parse_html_tags(text, keys=["plan"])
        assert "Step 1" in result["plan"]
        assert "Step 2" in result["plan"]
        assert "Step 3" in result["plan"]

    def test_parse_multiple_same_tags(self) -> None:
        """Test parsing multiple instances of the same tag."""
        text = "<memory>First memory</memory><memory>Second memory</memory>"
        result = parse_html_tags(text, keys=[], optional_keys=["memory"])
        # Should join multiple matches with newlines
        assert "First memory" in result["memory"]
        assert "Second memory" in result["memory"]

    def test_parse_empty_tag(self) -> None:
        """Test parsing empty tag content."""
        text = "<think></think>"
        result = parse_html_tags(text, keys=["think"])
        assert result["think"] == ""

    def test_parse_with_surrounding_text(self) -> None:
        """Test parsing tags with surrounding text."""
        text = "Some text before <think>My thought</think> and after"
        result = parse_html_tags(text, keys=["think"])
        assert result["think"] == "My thought"


# ============================================================================
# Test PromptElement
# ============================================================================


class TestPromptElement:
    """Tests for the PromptElement base class."""

    def test_visible_true(self) -> None:
        """Test element visibility when visible=True."""
        element = PromptElement(visible=True)
        assert element.visible is True

    def test_visible_false(self) -> None:
        """Test element visibility when visible=False."""
        element = PromptElement(visible=False)
        assert element.visible is False

    def test_visible_callable_true(self) -> None:
        """Test element visibility with callable returning True."""
        element = PromptElement(visible=lambda: True)
        assert element.visible is True

    def test_visible_callable_false(self) -> None:
        """Test element visibility with callable returning False."""
        element = PromptElement(visible=lambda: False)
        assert element.visible is False

    def test_prompt_hidden_when_not_visible(self) -> None:
        """Test that prompt returns empty string when not visible."""
        element = PromptElement(visible=False)
        assert element.prompt == ""

    def test_abstract_ex_hidden_when_not_visible(self) -> None:
        """Test that abstract_ex returns empty string when not visible."""
        element = PromptElement(visible=False)
        assert element.abstract_ex == ""

    def test_concrete_ex_hidden_when_not_visible(self) -> None:
        """Test that concrete_ex returns empty string when not visible."""
        element = PromptElement(visible=False)
        assert element.concrete_ex == ""

    def test_parse_answer_returns_empty_dict(self) -> None:
        """Test that base parse_answer returns empty dict."""
        element = PromptElement()
        assert element.parse_answer("any text") == {}


# ============================================================================
# Test Trunkater
# ============================================================================


class TestTrunkater:
    """Tests for the Trunkater class."""

    def test_content_visible(self) -> None:
        """Test content returns value when visible."""
        trunkater = Trunkater(content="Hello", visible=True)
        assert trunkater.content == "Hello"

    def test_content_hidden(self) -> None:
        """Test content returns empty when not visible."""
        trunkater = Trunkater(content="Hello", visible=False)
        assert trunkater.content == ""

    def test_shrink_before_start_iteration(self) -> None:
        """Test that shrink doesn't truncate before start iteration."""
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        trunkater = Trunkater(content=content, start_trunkate_iteration=3)

        # Call shrink twice (before iteration 3)
        trunkater.shrink()
        trunkater.shrink()

        assert trunkater.content == content
        assert trunkater.shrink_calls == 2

    def test_shrink_after_start_iteration(self) -> None:
        """Test that shrink truncates after start iteration."""
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7\nLine 8\nLine 9\nLine 10"
        trunkater = Trunkater(content=content, start_trunkate_iteration=2, shrink_speed=0.3)

        # Call shrink 3 times to trigger truncation
        trunkater.shrink()
        trunkater.shrink()
        trunkater.shrink()

        assert trunkater.shrink_calls == 3
        assert "Deleted" in trunkater.content
        assert trunkater.deleted_lines > 0

    def test_shrink_speed(self) -> None:
        """Test shrink speed affects truncation amount."""
        content = "\n".join([f"Line {i}" for i in range(100)])
        trunkater = Trunkater(content=content, start_trunkate_iteration=0, shrink_speed=0.5)

        trunkater.shrink()

        # Should have deleted about 50% of lines
        assert trunkater.deleted_lines >= 45
        assert trunkater.deleted_lines <= 55


# ============================================================================
# Test ShrinkableObservation
# ============================================================================


class TestShrinkableObservation:
    """Tests for the ShrinkableObservation class."""

    def test_prompt_with_html(self) -> None:
        """Test observation prompt includes HTML when enabled."""
        flags = ObsFlags(use_html=True, use_ax_tree=False)
        obs = ShrinkableObservation(
            html_content="<div>Hello</div>",
            axtree_content=None,
            error_content=None,
            focused_element=None,
            tabs_content=None,
            flags=flags,
        )
        prompt = obs.prompt
        assert "## HTML:" in prompt
        assert "<div>Hello</div>" in prompt

    def test_prompt_with_axtree(self) -> None:
        """Test observation prompt includes AXTree when enabled."""
        flags = ObsFlags(use_html=False, use_ax_tree=True)
        obs = ShrinkableObservation(
            html_content=None,
            axtree_content="[1] button 'Click me'",
            error_content=None,
            focused_element=None,
            tabs_content=None,
            flags=flags,
        )
        prompt = obs.prompt
        assert "## AXTree:" in prompt
        assert "[bid]" in prompt  # Should include bid note
        assert "[1] button 'Click me'" in prompt

    def test_prompt_with_tabs(self) -> None:
        """Test observation prompt includes tabs when enabled."""
        flags = ObsFlags(use_tabs=True)
        obs = ShrinkableObservation(
            html_content=None,
            axtree_content=None,
            error_content=None,
            focused_element=None,
            tabs_content="Tab 1: Google\nTab 2: GitHub",
            flags=flags,
        )
        prompt = obs.prompt
        assert "## Currently open tabs:" in prompt
        assert "Tab 1: Google" in prompt

    def test_prompt_with_error(self) -> None:
        """Test observation prompt includes error when enabled."""
        flags = ObsFlags(use_error_logs=True)
        obs = ShrinkableObservation(
            html_content=None,
            axtree_content=None,
            error_content="Element not found",
            focused_element=None,
            tabs_content=None,
            flags=flags,
        )
        prompt = obs.prompt
        assert "## Error from previous action:" in prompt
        assert "Element not found" in prompt

    def test_prompt_with_focused_element(self) -> None:
        """Test observation prompt includes focused element when enabled."""
        flags = ObsFlags(use_focused_element=True)
        obs = ShrinkableObservation(
            html_content=None,
            axtree_content=None,
            error_content=None,
            focused_element="btn123",
            tabs_content=None,
            flags=flags,
        )
        prompt = obs.prompt
        assert "## Focused element:" in prompt
        assert "btn123" in prompt

    def test_prompt_focused_element_none(self) -> None:
        """Test observation prompt shows None for no focused element."""
        flags = ObsFlags(use_focused_element=True)
        obs = ShrinkableObservation(
            html_content=None,
            axtree_content=None,
            error_content=None,
            focused_element=None,
            tabs_content=None,
            flags=flags,
        )
        prompt = obs.prompt
        assert "## Focused element:" in prompt
        assert "None" in prompt

    def test_prompt_axtree_with_coords_center(self) -> None:
        """Test AXTree includes center coords note."""
        flags = ObsFlags(use_ax_tree=True, extract_coords="center")
        obs = ShrinkableObservation(
            html_content=None,
            axtree_content="[1] button",
            error_content=None,
            focused_element=None,
            tabs_content=None,
            flags=flags,
        )
        prompt = obs.prompt
        assert "center coordinates" in prompt

    def test_prompt_axtree_with_coords_box(self) -> None:
        """Test AXTree includes bounding box coords note."""
        flags = ObsFlags(use_ax_tree=True, extract_coords="box")
        obs = ShrinkableObservation(
            html_content=None,
            axtree_content="[1] button",
            error_content=None,
            focused_element=None,
            tabs_content=None,
            flags=flags,
        )
        prompt = obs.prompt
        assert "bounding box" in prompt

    def test_shrink_calls_trunkaters(self) -> None:
        """Test shrink calls shrink on HTML and AXTree trunkaters."""
        flags = ObsFlags(use_html=True, use_ax_tree=True)
        obs = ShrinkableObservation(
            html_content="HTML content",
            axtree_content="AXTree content",
            error_content=None,
            focused_element=None,
            tabs_content=None,
            flags=flags,
        )
        obs.shrink()
        assert obs.html.shrink_calls == 1
        assert obs.axtree.shrink_calls == 1


# ============================================================================
# Test Think Element
# ============================================================================


class TestThink:
    """Tests for the Think prompt element."""

    def test_prompt_instructs_thinking(self) -> None:
        """Test that Think prompt instructs to reason before acting."""
        think = Think(visible=True)
        prompt = think.prompt
        assert "think" in prompt.lower()
        assert "action" in prompt.lower() or "step" in prompt.lower()

    def test_examples_no_think_tags(self) -> None:
        """Test that Think examples don't require <think> tag format."""
        think = Think(visible=True)
        assert "<think>" not in think.abstract_ex
        assert "<think>" not in think.concrete_ex

    def test_parse_answer_with_think_tags(self) -> None:
        """Test parsing when model produces <think> tags in text."""
        think = Think()
        result = think.parse_answer("<think>My thought process</think>")
        assert result["think"] == "My thought process"

    def test_parse_answer_no_tags(self) -> None:
        """Test parsing when model doesn't produce <think> tags (reasoning via native mechanism)."""
        think = Think()
        result = think.parse_answer("No think tag here")
        assert result == {}


# ============================================================================
# Test Plan Element
# ============================================================================


class TestPlan:
    """Tests for the Plan prompt element."""

    def test_prompt_includes_previous_plan(self) -> None:
        """Test prompt includes previous plan and step."""
        plan = Plan(previous_plan="1. Do A\n2. Do B", plan_step=1, visible=True)
        prompt = plan.prompt
        assert "step 1" in prompt
        assert "1. Do A" in prompt
        assert "2. Do B" in prompt

    def test_abstract_ex(self) -> None:
        """Test abstract example content."""
        plan = Plan(previous_plan="", plan_step=0, visible=True)
        assert "<plan>" in plan.abstract_ex
        assert "<step>" in plan.abstract_ex

    def test_parse_answer_with_plan_and_step(self) -> None:
        """Test parsing plan and step tags."""
        plan = Plan(previous_plan="", plan_step=0)
        result = plan.parse_answer("<plan>My plan</plan><step>3</step>")
        assert result["plan"] == "My plan"
        assert result["step"] == 3  # Should be converted to int

    def test_parse_answer_step_non_integer(self) -> None:
        """Test parsing step with non-integer value."""
        plan = Plan(previous_plan="", plan_step=0)
        result = plan.parse_answer("<plan>Plan</plan><step>not a number</step>")
        assert result["step"] == "not a number"  # Kept as string


# ============================================================================
# Test Memory Element
# ============================================================================


class TestMemory:
    """Tests for the Memory prompt element."""

    def test_abstract_ex(self) -> None:
        """Test abstract example content."""
        memory = Memory(visible=True)
        assert "<memory>" in memory.abstract_ex
        assert "remember" in memory.abstract_ex.lower()

    def test_concrete_ex(self) -> None:
        """Test concrete example content."""
        memory = Memory(visible=True)
        assert "<memory>" in memory.concrete_ex

    def test_parse_answer(self) -> None:
        """Test parsing memory tag."""
        memory = Memory()
        result = memory.parse_answer("<memory>Important info</memory>")
        assert result["memory"] == "Important info"

    def test_parse_answer_missing(self) -> None:
        """Test parsing when memory tag is missing."""
        memory = Memory()
        result = memory.parse_answer("No memory tag")
        assert "memory" not in result


# ============================================================================
# Test Criticise Element
# ============================================================================


class TestCriticise:
    """Tests for the Criticise prompt element."""

    def test_abstract_ex(self) -> None:
        """Test abstract example content."""
        criticise = Criticise(visible=True)
        assert "<action_draft>" in criticise.abstract_ex
        assert "<criticise>" in criticise.abstract_ex

    def test_concrete_ex(self) -> None:
        """Test concrete example content."""
        criticise = Criticise(visible=True)
        assert "<action_draft>" in criticise.concrete_ex
        assert "<criticise>" in criticise.concrete_ex

    def test_parse_answer(self) -> None:
        """Test parsing action_draft and criticise tags."""
        criticise = Criticise()
        result = criticise.parse_answer("<action_draft>Click btn</action_draft><criticise>Might fail</criticise>")
        assert result["action_draft"] == "Click btn"
        assert result["criticise"] == "Might fail"


# ============================================================================
# Test BeCautious Element
# ============================================================================


class TestBeCautious:
    """Tests for the BeCautious prompt element."""

    def test_prompt_content(self) -> None:
        """Test prompt contains caution instructions."""
        cautious = BeCautious(visible=True)
        assert "cautious" in cautious.prompt.lower()
        assert "verify" in cautious.prompt.lower()


# ============================================================================
# Test Hints Element
# ============================================================================


class TestHints:
    """Tests for the Hints prompt element."""

    def test_prompt_content(self) -> None:
        """Test prompt contains helpful hints."""
        hints = Hints(visible=True)
        prompt = hints.prompt
        assert "Note:" in prompt
        assert "bid" in prompt
        assert "combobox" in prompt or "dropdown" in prompt


# ============================================================================
# Test HistoryStep
# ============================================================================


class TestHistoryStep:
    """Tests for the HistoryStep class."""

    def test_prompt_with_action(self) -> None:
        """Test step prompt includes action when enabled."""
        flags = ObsFlags(use_action_history=True)
        step = HistoryStep(action="click('btn')", memory=None, thought=None, error=None, flags=flags)
        assert "<action>" in step.prompt
        assert "click('btn')" in step.prompt

    def test_prompt_with_thought(self) -> None:
        """Test step prompt includes thought when enabled."""
        flags = ObsFlags(use_think_history=True)
        step = HistoryStep(action=None, memory=None, thought="My thought", error=None, flags=flags)
        assert "<think>" in step.prompt
        assert "My thought" in step.prompt

    def test_prompt_with_memory(self) -> None:
        """Test step prompt includes memory."""
        flags = ObsFlags()
        step = HistoryStep(action=None, memory="Remember this", thought=None, error=None, flags=flags)
        assert "<memory>" in step.prompt
        assert "Remember this" in step.prompt

    def test_prompt_with_error(self) -> None:
        """Test step prompt includes error when enabled."""
        flags = ObsFlags(use_error_logs=True, use_past_error_logs=True)
        step = HistoryStep(action=None, memory=None, thought=None, error="Error occurred", flags=flags)
        assert "Error" in step.prompt
        assert "Error occurred" in step.prompt

    def test_prompt_no_thought_placeholder(self) -> None:
        """Test step shows explicit placeholder when thought is absent."""
        flags = ObsFlags(use_think_history=True)
        step = HistoryStep(action=None, memory=None, thought=None, error=None, flags=flags)
        assert "[No thinking on this step]" in step.prompt
        assert "<think>" not in step.prompt

    def test_prompt_renders_none_action_when_history_enabled(self) -> None:
        """Test step renders action even when None (action is always shown)."""
        flags = ObsFlags(use_action_history=True)
        step = HistoryStep(action=None, memory=None, thought=None, error=None, flags=flags)
        assert "<action>" in step.prompt

    def test_prompt_empty_when_nothing_enabled(self) -> None:
        """Test step prompt is empty when nothing is enabled."""
        flags = ObsFlags(use_action_history=False, use_think_history=False)
        step = HistoryStep(action="action", memory=None, thought="thought", error=None, flags=flags)
        assert step.prompt == ""


# ============================================================================
# Test History
# ============================================================================


class TestHistory:
    """Tests for the History class."""

    def test_prompt_with_history_enabled(self) -> None:
        """Test history prompt when use_history is True."""
        flags = ObsFlags(use_history=True, use_action_history=True)
        history = History(
            obs_history=["obs1", "obs2"],
            actions=["action1", "action2"],
            memories=[None, None],
            thoughts=[None, None],
            errors=[None, None],
            flags=flags,
        )
        prompt = history.prompt
        assert "# History of interaction" in prompt
        assert "## step 0" in prompt
        assert "## step 1" in prompt

    def test_prompt_empty_when_disabled(self) -> None:
        """Test history prompt is empty when use_history is False."""
        flags = ObsFlags(use_history=False)
        history = History(
            obs_history=["obs1"],
            actions=["action1"],
            memories=[None],
            thoughts=[None],
            errors=[None],
            flags=flags,
        )
        assert history.prompt == ""

    def test_prompt_empty_when_no_steps(self) -> None:
        """Test history prompt is empty when no history steps."""
        flags = ObsFlags(use_history=True)
        history = History(
            obs_history=[],
            actions=[],
            memories=[],
            thoughts=[],
            errors=[],
            flags=flags,
        )
        assert history.prompt == ""


# ============================================================================
# Test ObsFlags
# ============================================================================


class TestObsFlags:
    """Tests for the ObsFlags configuration class."""

    def test_default_values(self) -> None:
        """Test default flag values."""
        flags = ObsFlags()
        assert flags.use_html is True
        assert flags.use_ax_tree is False
        assert flags.use_screenshot is True
        assert flags.use_history is False

    def test_custom_values(self) -> None:
        """Test custom flag values."""
        flags = ObsFlags(use_html=False, use_ax_tree=True, use_screenshot=False)
        assert flags.use_html is False
        assert flags.use_ax_tree is True
        assert flags.use_screenshot is False


# ============================================================================
# Test GenericPromptFlags
# ============================================================================


class TestGenericPromptFlags:
    """Tests for the GenericPromptFlags configuration class."""

    def test_default_values(self) -> None:
        """Test default flag values."""
        flags = GenericPromptFlags()
        assert flags.use_plan is False
        assert flags.use_thinking is False
        assert flags.use_memory is False
        assert flags.use_concrete_example is True

    def test_nested_obs_flags(self) -> None:
        """Test nested ObsFlags."""
        flags = GenericPromptFlags(obs=ObsFlags(use_html=False))
        assert flags.obs.use_html is False


# ============================================================================
# Test MainPrompt
# ============================================================================


class TestMainPrompt:
    """Tests for the MainPrompt class."""

    @pytest.fixture
    def basic_main_prompt(self) -> MainPrompt:
        """Basic MainPrompt for testing."""
        flags = GenericPromptFlags(
            use_thinking=False,
            use_plan=False,
            use_memory=False,
            use_criticise=False,
            use_concrete_example=False,
            use_abstract_example=False,
        )
        return MainPrompt(
            obs_history=[],
            actions=[],
            memories=[],
            thoughts=[],
            errors=[],
            previous_plan="",
            step=0,
            flags=flags,
            goal="Click the button",
            current_obs_components={"html": "<button>Click me</button>"},
        )

    def test_build_prompt_includes_goal(self, basic_main_prompt: MainPrompt) -> None:
        """Test that built prompt includes the goal."""
        prompt = basic_main_prompt.build_prompt()
        assert "Click the button" in prompt

    def test_build_prompt_includes_instructions(self, basic_main_prompt: MainPrompt) -> None:
        """Test that built prompt includes instructions."""
        prompt = basic_main_prompt.build_prompt()
        assert "# Instructions" in prompt

    def test_build_prompt_includes_observation(self, basic_main_prompt: MainPrompt) -> None:
        """Test that built prompt includes observation."""
        prompt = basic_main_prompt.build_prompt()
        assert "# Observation" in prompt

    def test_build_prompt_with_thinking(self) -> None:
        """Test prompt includes thinking instruction when enabled."""
        flags = GenericPromptFlags(use_thinking=True)
        prompt = MainPrompt(
            obs_history=[],
            actions=[],
            memories=[],
            thoughts=[],
            errors=[],
            previous_plan="",
            step=0,
            flags=flags,
            goal="Goal",
        )
        built = prompt.build_prompt()
        assert "think" in built.lower()
        assert "action" in built.lower() or "step" in built.lower()

    def test_build_prompt_with_plan(self) -> None:
        """Test prompt includes plan when enabled."""
        flags = GenericPromptFlags(use_plan=True)
        prompt = MainPrompt(
            obs_history=[],
            actions=[],
            memories=[],
            thoughts=[],
            errors=[],
            previous_plan="Previous plan content",
            step=1,
            flags=flags,
            goal="Goal",
        )
        built = prompt.build_prompt()
        assert "# Plan:" in built
        assert "Previous plan content" in built

    def test_build_prompt_chat_mode(self) -> None:
        """Test prompt in chat mode."""
        flags = GenericPromptFlags(enable_chat=True)
        prompt = MainPrompt(
            obs_history=[],
            actions=[],
            memories=[],
            thoughts=[],
            errors=[],
            previous_plan="",
            step=0,
            flags=flags,
            goal="Chat goal",
        )
        built = prompt.build_prompt()
        assert "UI Assistant" in built
        assert "## Chat messages:" in built

    def test_build_prompt_with_extra_instructions(self) -> None:
        """Test prompt includes extra instructions."""
        flags = GenericPromptFlags(extra_instructions="Be careful with forms")
        prompt = MainPrompt(
            obs_history=[],
            actions=[],
            memories=[],
            thoughts=[],
            errors=[],
            previous_plan="",
            step=0,
            flags=flags,
            goal="Goal",
        )
        built = prompt.build_prompt()
        assert "## Extra instructions:" in built
        assert "Be careful with forms" in built

    def test_parse_answer(self, basic_main_prompt: MainPrompt) -> None:
        """Test parse_answer extracts all tags."""
        text = "<think>Thought</think><plan>Plan</plan><memory>Memory</memory>"
        result = basic_main_prompt.parse_answer(text)
        # Think is parsed specially (required)
        assert "think" in result

    def test_shrink_calls_observation_shrink(self) -> None:
        """Test shrink calls observation shrink."""
        flags = GenericPromptFlags()
        prompt = MainPrompt(
            obs_history=[],
            actions=[],
            memories=[],
            thoughts=[],
            errors=[],
            previous_plan="",
            step=0,
            flags=flags,
            goal="Goal",
            current_obs_components={"html": "HTML content"},
        )
        assert prompt.observation is not None
        prompt.shrink()
        assert prompt.observation.html.shrink_calls == 1


# ============================================================================
# Test GenericAgentConfig
# ============================================================================


class TestGenericAgentConfig:
    """Tests for the GenericAgentConfig class."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = GenericAgentConfig(llm_config=LLMConfig(model_name="test-model"))
        assert config.max_retry == 4
        assert config.max_actions == 50
        assert "agent" in config.system_prompt.lower()

    def test_make_creates_agent(self) -> None:
        """Test make() creates a GenericAgent."""
        config = GenericAgentConfig(llm_config=LLMConfig(model_name="test-model"))
        action_set = [ActionSchema(name="click", description="Click", parameters={})]
        agent = config.make(action_set=action_set)
        assert agent.config == config
        assert len(agent.action_set) == 2  # click + stop action


# ============================================================================
# Test GenericAgent Observation Extraction
# ============================================================================


class TestGenericAgentObservationExtraction:
    """Tests for GenericAgent observation extraction methods."""

    @pytest.fixture
    def agent(self) -> GenericAgent:
        """Create a GenericAgent for testing."""
        config = GenericAgentConfig(llm_config=LLMConfig(model_name="test-model"))
        return GenericAgent(config=config, action_set=[])

    def test_extract_goal_from_named_content(self, agent) -> None:
        """Test goal extraction from content named 'goal'."""
        obs = Observation(contents=[Content.from_data("Click the button", name="goal")])
        goal = agent._extract_goal(obs)
        assert goal == "Click the button"

    def test_extract_goal_from_unnamed_short_content(self, agent) -> None:
        """Test goal extraction from unnamed short content."""
        obs = Observation(contents=[Content.from_data("Short task description")])
        goal = agent._extract_goal(obs)
        assert goal == "Short task description"

    def test_extract_goal_default(self, agent) -> None:
        """Test default goal when no suitable content found."""
        img = Image.new("RGB", (10, 10))
        obs = Observation(contents=[Content.from_data(img, name="screenshot")])
        goal = agent._extract_goal(obs)
        assert goal == "Complete the task."

    def test_extract_error_from_observation(self, agent) -> None:
        """Test error extraction from observation."""
        obs = Observation(contents=[Content.from_data("Element not found", name="error")])
        error = agent._extract_error(obs)
        assert error == "Element not found"

    def test_extract_error_none_when_missing(self, agent) -> None:
        """Test error extraction returns None when no error."""
        obs = Observation(contents=[Content.from_data("Some content", name="html")])
        error = agent._extract_error(obs)
        assert error is None

    def test_extract_obs_data(self, agent) -> None:
        """Test extraction of observation data (text and components)."""
        obs = Observation(
            contents=[
                Content.from_data("<html>content</html>", name="pruned_html"),
                Content.from_data("[1] button", name="axtree_txt"),
                Content.from_data("Tab 1", name="tabs"),
            ]
        )
        _obs_text, components = agent._extract_obs_data(obs)
        assert components["html"] == "<html>content</html>"
        assert components["axtree"] == "[1] button"
        assert components["tabs"] == "Tab 1"

    def test_extract_screenshots(self, agent) -> None:
        """Test screenshot extraction."""
        img = Image.new("RGB", (100, 100))
        obs = Observation(
            contents=[
                Content.from_data("text"),
                Content.from_data(img, name="screenshot"),
            ]
        )
        screenshots = agent._extract_screenshots(obs)
        assert len(screenshots) == 1
        assert screenshots[0] == img

    def test_extract_screenshots_disabled(self, agent) -> None:
        """Test screenshot extraction when disabled."""
        agent.config.flags.obs.use_screenshot = False
        img = Image.new("RGB", (100, 100))
        obs = Observation(contents=[Content.from_data(img, name="screenshot")])
        screenshots = agent._extract_screenshots(obs)
        assert screenshots == []
