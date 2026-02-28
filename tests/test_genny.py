"""Tests for Genny.

Most tests do NOT require LLM calls — they exercise pure functions and agent
state manipulation directly. LLM-touching paths (_summarize_past, _act_pass,
step) use MagicMock so the test suite stays fast.
"""

from unittest.mock import MagicMock

from agentlab2.agents.genny import (
    Genny,
    GennyConfig,
    NativeToolAdapter,
    TextToolAdapter,
    _format_action_list,
    _format_summaries_block,
    _format_tools_as_text,
    _json_type_to_python,
    _obs_section_header,
    _truncate_message,
)
from agentlab2.core import Action, ActionSchema, Observation
from agentlab2.llm import LLMConfig, LLMResponse, Usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_schema(name: str = "click", description: str = "Click an element.") -> ActionSchema:
    return ActionSchema(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "element_id": {"type": "string", "description": "The element id."},
                "force": {"type": "boolean"},
            },
            "required": ["element_id"],
        },
    )


def _make_agent(
    render_last_n_obs: int | None = None,
    enable_summarize: bool = False,
    summarize_cot_only: bool = False,
    tools_as_text: bool = False,
) -> Genny:
    config = GennyConfig(
        llm_config=LLMConfig(model_name="test"),
        render_last_n_obs=render_last_n_obs,
        enable_summarize=enable_summarize,
        summarize_cot_only=summarize_cot_only,
        tools_as_text=tools_as_text,
    )
    return Genny(config=config, action_schemas=[_make_schema()])


def _simulate_steps(agent: Genny, n_steps: int) -> None:
    """Populate agent state as if n_steps obs+asst pairs have been processed."""
    agent.goal = [{"role": "user", "content": "goal"}]
    for i in range(n_steps):
        agent.history.append([{"role": "user", "content": f"obs_{i}"}])
        agent.history.append([{"role": "assistant", "content": f"asst_{i}"}])
    # Final pending obs (no asst response yet)
    agent.history.append([{"role": "user", "content": f"obs_{n_steps}"}])


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestJsonTypeToPython:
    def test_known_types(self) -> None:
        assert _json_type_to_python("string") == "str"
        assert _json_type_to_python("integer") == "int"
        assert _json_type_to_python("boolean") == "bool"
        assert _json_type_to_python("array") == "list"

    def test_unknown_falls_back_to_any(self) -> None:
        assert _json_type_to_python("unknown") == "Any"


class TestFormatSummariesBlock:
    def test_has_header(self) -> None:
        result = _format_summaries_block(["s1", "s2"])
        assert result.startswith("## Summary of past interactions")

    def test_all_summaries_included(self) -> None:
        result = _format_summaries_block(["alpha", "beta"])
        assert "alpha" in result
        assert "beta" in result

    def test_step_headers_present(self) -> None:
        result = _format_summaries_block(["first", "second"])
        assert "### Step 1" in result
        assert "### Step 2" in result

    def test_single_summary(self) -> None:
        result = _format_summaries_block(["only one"])
        assert "only one" in result
        assert "### Step 1" in result


class TestObsSectionHeader:
    def test_none_produces_generic_header(self) -> None:
        assert "observations" in _obs_section_header(None).lower()

    def test_n_included_in_header(self) -> None:
        assert "3" in _obs_section_header(3)


class TestTruncateMessage:
    def test_truncates_long_content(self) -> None:
        msg = {"role": "user", "content": "a" * 200}
        result = _truncate_message(msg, max_chars=50)
        assert len(result["content"]) < 200
        assert "truncated" in result["content"]

    def test_short_content_unchanged(self) -> None:
        msg = {"role": "user", "content": "hello"}
        assert _truncate_message(msg, max_chars=100) == msg

    def test_non_string_content_unchanged(self) -> None:
        msg = {"role": "user", "content": [{"type": "image_url"}]}
        assert _truncate_message(msg, max_chars=10) == msg


class TestFormatToolsAsText:
    def test_contains_function_name(self) -> None:
        schema = _make_schema("browser_click")
        result = _format_tools_as_text([schema])
        assert "def browser_click(" in result

    def test_required_arg_has_no_default(self) -> None:
        schema = _make_schema()
        result = _format_tools_as_text([schema])
        assert "element_id: str" in result
        # element_id is required — no "= None"
        assert "element_id: str = None" not in result

    def test_optional_arg_has_default(self) -> None:
        schema = _make_schema()
        result = _format_tools_as_text([schema])
        # force is not required
        assert "force: bool = None" in result

    def test_includes_tool_call_instruction(self) -> None:
        result = _format_tools_as_text([_make_schema()])
        assert "<tool_call>" in result

    def test_multiple_tools(self) -> None:
        schemas = [_make_schema("click"), _make_schema("type")]
        result = _format_tools_as_text(schemas)
        assert "def click(" in result
        assert "def type(" in result


# ---------------------------------------------------------------------------
# NativeToolAdapter
# ---------------------------------------------------------------------------


class TestNativeToolAdapter:
    def test_encode_returns_tool_dicts(self) -> None:
        adapter = NativeToolAdapter()
        schema = _make_schema()
        tools, msgs = adapter.encode([schema], [{"role": "user", "content": "hi"}])
        assert tools == [schema.as_dict()]
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_decode_parses_tool_calls(self) -> None:
        adapter = NativeToolAdapter()
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "click"
        tc.function.arguments = '{"element_id": "btn"}'
        response = MagicMock()
        response.tool_calls = [tc]
        actions = adapter.decode(response)
        assert len(actions) == 1
        assert actions[0].name == "click"
        assert actions[0].arguments == {"element_id": "btn"}

    def test_decode_empty_when_no_tool_calls(self) -> None:
        adapter = NativeToolAdapter()
        response = MagicMock()
        response.tool_calls = None
        assert adapter.decode(response) == []


# ---------------------------------------------------------------------------
# TextToolAdapter
# ---------------------------------------------------------------------------


class TestTextToolAdapter:
    def test_encode_injects_sigs_into_system(self) -> None:
        adapter = TextToolAdapter()
        schema = _make_schema("click")
        messages = [{"role": "system", "content": "You are an agent."}]
        _, result = adapter.encode([schema], messages)
        assert "def click(" in result[0]["content"]
        assert result[0]["role"] == "system"

    def test_encode_returns_empty_tools(self) -> None:
        adapter = TextToolAdapter()
        api_tools, _ = adapter.encode([_make_schema()], [{"role": "system", "content": "s"}])
        assert api_tools == []

    def test_encode_no_tools_returns_messages_unchanged(self) -> None:
        adapter = TextToolAdapter()
        messages = [{"role": "user", "content": "hi"}]
        tools, result = adapter.encode([], messages)
        assert tools == []
        assert result == messages

    def test_encode_does_not_mutate_original_messages(self) -> None:
        adapter = TextToolAdapter()
        original = [{"role": "system", "content": "original"}]
        adapter.encode([_make_schema()], original)
        assert original[0]["content"] == "original"

    def test_decode_parses_tool_call_tags(self) -> None:
        adapter = TextToolAdapter()
        response = MagicMock()
        response.content = 'Reasoning...\n<tool_call>{"name": "click", "arguments": {"element_id": "btn"}}</tool_call>'
        actions = adapter.decode(response)
        assert len(actions) == 1
        assert actions[0].name == "click"
        assert actions[0].arguments == {"element_id": "btn"}

    def test_decode_multiple_calls(self) -> None:
        adapter = TextToolAdapter()
        response = MagicMock()
        response.content = (
            '<tool_call>{"name": "click", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "type", "arguments": {"text": "hi"}}</tool_call>'
        )
        actions = adapter.decode(response)
        assert len(actions) == 2
        assert actions[0].name == "click"
        assert actions[1].name == "type"

    def test_decode_empty_when_no_tags(self) -> None:
        adapter = TextToolAdapter()
        response = MagicMock()
        response.content = "Just thinking aloud."
        assert adapter.decode(response) == []

    def test_decode_skips_malformed_json(self) -> None:
        adapter = TextToolAdapter()
        response = MagicMock()
        response.content = "<tool_call>NOT JSON</tool_call>"
        assert adapter.decode(response) == []


# ---------------------------------------------------------------------------
# Genny state — no LLM required
# ---------------------------------------------------------------------------


class TestIngestObs:
    def test_first_obs_sets_goal(self) -> None:
        agent = _make_agent()
        agent._ingest_obs([{"role": "user", "content": "goal text"}])
        assert agent.goal == [{"role": "user", "content": "goal text"}]
        assert agent.history == []

    def test_first_obs_with_extra_messages_puts_rest_in_history(self) -> None:
        agent = _make_agent()
        agent._ingest_obs([
            {"role": "user", "content": "goal"},
            {"role": "user", "content": "screenshot"},
        ])
        assert agent.goal == [{"role": "user", "content": "goal"}]
        assert agent.history == [[{"role": "user", "content": "screenshot"}]]

    def test_subsequent_obs_appended_as_group(self) -> None:
        agent = _make_agent()
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._ingest_obs([{"role": "user", "content": "obs_1"}])
        assert len(agent.history) == 1
        assert agent.history[0] == [{"role": "user", "content": "obs_1"}]


class TestWindowedHistory:
    def test_none_returns_all(self) -> None:
        agent = _make_agent(render_last_n_obs=None)
        _simulate_steps(agent, n_steps=3)
        flat = agent._windowed_history()
        expected = [msg for group in agent.history for msg in group]
        assert flat == expected

    def test_render_last_1_obs(self) -> None:
        agent = _make_agent(render_last_n_obs=1)
        _simulate_steps(agent, n_steps=3)
        flat = agent._windowed_history()
        # Only the last obs group — no preceding asst group (its action is in summaries)
        assert len(flat) == 1
        assert flat[0]["content"] == "obs_3"

    def test_render_last_2_obs(self) -> None:
        agent = _make_agent(render_last_n_obs=2)
        _simulate_steps(agent, n_steps=3)
        flat = agent._windowed_history()
        # Only the last 2 obs groups — asst groups excluded
        assert len(flat) == 2
        assert flat[0]["content"] == "obs_2"
        assert flat[1]["content"] == "obs_3"

    def test_window_larger_than_history_returns_all_obs(self) -> None:
        agent = _make_agent(render_last_n_obs=100)
        _simulate_steps(agent, n_steps=2)
        flat = agent._windowed_history()
        # All obs groups (obs_0, obs_1, obs_2) — asst groups excluded
        obs_groups = [g for g in agent.history if g and g[0].get("role") != "assistant"]
        assert flat == [msg for group in obs_groups for msg in group]

    def test_tool_messages_stripped_from_windowed_obs(self) -> None:
        agent = _make_agent(render_last_n_obs=1)
        agent.goal = [{"role": "user", "content": "goal"}]
        # Simulate: one completed step (obs with tool result + user content) + asst + pending obs
        agent.history.append([{"role": "tool", "content": "Success", "tool_call_id": "c1"}, {"role": "user", "content": "axtree"}])
        agent.history.append([{"role": "assistant", "content": "act", "tool_calls": [{"id": "c1"}]}])
        agent.history.append([{"role": "tool", "content": "Success", "tool_call_id": "c2"}, {"role": "user", "content": "current_axtree"}])
        flat = agent._windowed_history()
        # Tool message stripped; only the user content from the latest obs remains
        assert all(m.get("role") != "tool" for m in flat)
        assert flat[0]["content"] == "current_axtree"


class TestChooseContext:
    def test_always_starts_with_system(self) -> None:
        agent = _make_agent()
        _simulate_steps(agent, n_steps=1)
        messages = agent._choose_context()
        assert messages[0]["role"] == "system"

    def test_goal_always_included(self) -> None:
        agent = _make_agent(render_last_n_obs=1)
        _simulate_steps(agent, n_steps=5)
        messages = agent._choose_context()
        assert any(m.get("content") == "goal" for m in messages)

    def test_past_summaries_collapsed_into_single_block(self) -> None:
        """All past summaries are combined into one assistant message with a header."""
        agent = _make_agent(enable_summarize=False)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["summary A", "summary B"]
        messages = agent._choose_context()
        asst_messages = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
        assert len(asst_messages) == 1
        block = asst_messages[0]["content"]
        assert "## Summary of past interactions" in block
        assert "summary A" in block
        assert "summary B" in block

    def test_current_step_summary_appears_after_obs_when_summarize_enabled(self) -> None:
        """When enable_summarize=True, the current step's summary (self.summaries[-1])
        is placed AFTER the windowed history, not before it."""
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["past summary", "current step summary"]
        agent.history = [[{"role": "user", "content": "latest obs"}]]
        messages = agent._choose_context()
        contents = [m.get("content", "") for m in messages if isinstance(m, dict)]
        obs_idx = next(i for i, c in enumerate(contents) if c == "latest obs")
        current_summary_idx = next(i for i, c in enumerate(contents) if c == "current step summary")
        assert current_summary_idx > obs_idx, "current step summary should appear after the obs"

    def test_obs_section_header_inserted_before_windowed_history(self) -> None:
        agent = _make_agent(render_last_n_obs=2)
        _simulate_steps(agent, n_steps=1)
        messages = agent._choose_context()
        user_contents = [m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "user"]
        assert any("most recent observations" in c for c in user_contents)

    def test_ends_with_react_prompt_when_summarize_disabled(self) -> None:
        agent = _make_agent(enable_summarize=False)
        _simulate_steps(agent, n_steps=1)
        messages = agent._choose_context()
        assert messages[-1]["content"] == agent.config.react_prompt

    def test_ends_with_act_prompt_when_summarize_enabled(self) -> None:
        agent = _make_agent(enable_summarize=True)
        _simulate_steps(agent, n_steps=1)
        messages = agent._choose_context()
        assert messages[-1]["content"] == agent.config.act_prompt


# ---------------------------------------------------------------------------
# LLM-dependent paths — mocked
# ---------------------------------------------------------------------------


def _mock_llm_response(text: str = "summary text") -> LLMResponse:
    from litellm import Message as LitellmMessage
    return LLMResponse(
        message=LitellmMessage(role="assistant", content=text),
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


class TestFormatActionList:
    def test_formats_single_action(self) -> None:
        actions = [Action(name="click", arguments={"bid": "btn1"})]
        assert "click(bid='btn1')" in _format_action_list(actions)

    def test_formats_multiple_actions(self) -> None:
        actions = [Action(name="click", arguments={}), Action(name="type", arguments={"text": "hi"})]
        result = _format_action_list(actions)
        assert "click()" in result
        assert "type(text='hi')" in result

    def test_empty_actions_returns_no_action(self) -> None:
        assert _format_action_list([]) == "no action"


class TestSummarizePast:
    def test_returns_summary_string(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response("my summary"))
        summary, _ = agent._summarize_past()
        assert summary == "my summary"

    def test_includes_prior_summaries_in_prompt(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["prior summary"]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response())
        agent._summarize_past()
        prompt = agent.summarize_llm.call_args[0][0]
        contents = [m.get("content", "") for m in prompt.messages if isinstance(m, dict)]
        assert any("prior summary" in c for c in contents)

    def test_cot_mode_uses_cot_prompt(self) -> None:
        agent = _make_agent(enable_summarize=True, summarize_cot_only=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response())
        agent._summarize_past()
        prompt = agent.summarize_llm.call_args[0][0]
        assert prompt.messages[-1]["content"] == agent.config.summarize_cot_prompt

    def test_verbose_mode_uses_verbose_prompt(self) -> None:
        agent = _make_agent(enable_summarize=True, summarize_cot_only=False)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response())
        agent._summarize_past()
        prompt = agent.summarize_llm.call_args[0][0]
        assert prompt.messages[-1]["content"] == agent.config.summarize_verbose_prompt

    def test_uses_same_system_prompt_as_act_pass(self) -> None:
        """Both passes start with the same system_prompt → cache hit on shared prefix."""
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response())
        agent._summarize_past()
        prompt = agent.summarize_llm.call_args[0][0]
        assert isinstance(prompt.messages[0], dict)
        assert prompt.messages[0]["content"] == agent.config.system_prompt

    def test_prior_summaries_formatted_as_block(self) -> None:
        """Prior summaries appear as a single block with step headers."""
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["step one cot\n\nAction: click()", "step two cot\n\nAction: type()"]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response())
        agent._summarize_past()
        prompt = agent.summarize_llm.call_args[0][0]
        asst_msgs = [m for m in prompt.messages if isinstance(m, dict) and m.get("role") == "assistant"]
        assert len(asst_msgs) == 1
        assert "### Step 1" in asst_msgs[0]["content"]
        assert "### Step 2" in asst_msgs[0]["content"]

    def test_uses_windowed_history_not_separate_obs(self) -> None:
        """_summarize_past reads obs from _windowed_history() (same as act pass)."""
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.history = [[{"role": "user", "content": "obs from history"}]]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response())
        agent._summarize_past()
        prompt = agent.summarize_llm.call_args[0][0]
        contents = [m.get("content", "") for m in prompt.messages if isinstance(m, dict)]
        assert any("obs from history" in c for c in contents)

    def test_passes_same_tools_as_act_pass_for_cache(self) -> None:
        """Summarize LLM receives the same tools as the act LLM → identical cache key prefix."""
        agent = _make_agent(enable_summarize=True)  # NativeToolAdapter (default)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response())
        agent._summarize_past()
        prompt = agent.summarize_llm.call_args[0][0]
        assert len(prompt.tools) > 0, "Summarize prompt must include tool definitions for cache parity"
        assert prompt.tools[0]["function"]["name"] == "click"

    def test_summarize_llm_uses_same_tool_choice_as_act_llm(self) -> None:
        """Summarize LLM config matches act LLM tool_choice for identical cache key."""
        agent = _make_agent(enable_summarize=True)
        assert agent._summarize_llm_config.tool_choice == agent.config.llm_config.tool_choice


class TestStep:
    def test_step_appends_summary_with_action_when_enabled(self) -> None:
        """Summary entry includes both the LLM reasoning and the decided action."""
        agent = _make_agent(enable_summarize=True)
        agent.llm = MagicMock(return_value=_mock_llm_response("action"))
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response("step summary"))
        obs = Observation.from_text("goal text")
        agent.step(obs)
        assert len(agent.summaries) == 1
        assert "step summary" in agent.summaries[0]
        assert "Action:" in agent.summaries[0]

    def test_step_extracts_cot_to_summaries_when_summarize_disabled(self) -> None:
        """When enable_summarize=False, act-pass COT is extracted into self.summaries."""
        agent = _make_agent(enable_summarize=False)
        agent.llm = MagicMock(return_value=_mock_llm_response("I think therefore I act"))
        obs = Observation.from_text("goal text")
        agent.step(obs)
        assert len(agent.summaries) == 1
        assert "I think therefore I act" in agent.summaries[0]
        assert "Action:" in agent.summaries[0]

    def test_step_increments_action_count(self) -> None:
        agent = _make_agent()
        agent.llm = MagicMock(return_value=_mock_llm_response())
        agent.step(Observation.from_text("goal"))
        assert agent._actions_cnt == 1

    def test_step_issues_stop_action_when_limit_reached(self) -> None:
        agent = _make_agent()
        agent.config = agent.config.model_copy(update={"max_actions": 0})
        result = agent.step(Observation.from_text("obs"))
        assert len(result.actions) == 1
        assert result.actions[0].name == "final_step"

    def test_thoughts_is_summary_when_summarize_enabled(self) -> None:
        """thoughts captures the summarize-pass COT, before the action is appended."""
        agent = _make_agent(enable_summarize=True)
        agent.llm = MagicMock(return_value=_mock_llm_response("act text"))
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response("my cot reasoning"))
        result = agent.step(Observation.from_text("goal text"))
        assert result.thoughts == "my cot reasoning"
        # The stored summary has the action appended, but thoughts is the raw COT.
        assert "Action:" not in result.thoughts

    def test_thoughts_is_inline_content_when_summarize_disabled(self) -> None:
        """thoughts captures the act-pass inline content when no summarize pass."""
        agent = _make_agent(enable_summarize=False)
        agent.llm = MagicMock(return_value=_mock_llm_response("I think therefore I act"))
        result = agent.step(Observation.from_text("goal text"))
        assert result.thoughts == "I think therefore I act"

    def test_thoughts_is_none_when_no_content(self) -> None:
        """thoughts is None when the LLM returns no text content."""
        agent = _make_agent(enable_summarize=False)
        agent.llm = MagicMock(return_value=_mock_llm_response(""))
        result = agent.step(Observation.from_text("goal text"))
        assert result.thoughts is None
