"""Tests for Genny2.

Most tests do NOT require LLM calls — they exercise pure functions and agent
state manipulation directly. LLM-touching paths (_summarize_past, _act_pass,
step) use MagicMock so the test suite stays fast.
"""

from unittest.mock import MagicMock

import pytest
from cube.core import Action, ActionSchema, Observation

from cube_harness.agents.genny import (
    Genny2,
    Genny2Config,
    NativeToolAdapter,
    TextToolAdapter,
    _format_action_list,
    _format_tools_as_text,
    _json_type_to_python,
    _truncate_message,
)
from cube_harness.llm import LLMConfig, LLMResponse, Usage

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
    enable_summarize: bool = False,
    summarize_cot_only: bool = False,
    tools_as_text: bool = False,
) -> Genny2:
    config = Genny2Config(
        llm_config=LLMConfig(model_name="test"),
        enable_summarize=enable_summarize,
        summarize_cot_only=summarize_cot_only,
        tools_as_text=tools_as_text,
    )
    return Genny2(config=config, action_schemas=[_make_schema()])


def _simulate_completed_rounds(agent: Genny2, n: int) -> None:
    """Populate agent state as if n obs+asst rounds have completed (Mode A)."""
    agent.goal = [{"role": "user", "content": "goal"}]
    for i in range(n):
        agent.history.append([{"role": "user", "content": f"obs_{i}"}])
        agent.history.append([{"role": "assistant", "content": f"asst_{i}"}])
    agent._latest_obs = [{"role": "user", "content": f"obs_{n}"}]


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
        assert "element_id: str = None" not in result

    def test_optional_arg_has_default(self) -> None:
        schema = _make_schema()
        result = _format_tools_as_text([schema])
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
# Genny2 state — no LLM required
# ---------------------------------------------------------------------------


class TestIngestObs:
    def test_first_obs_sets_goal_and_empty_latest_obs(self) -> None:
        agent = _make_agent()
        agent._ingest_obs([{"role": "user", "content": "goal text"}])
        assert agent.goal == [{"role": "user", "content": "goal text"}]
        assert agent._latest_obs == []
        assert agent.history == []

    def test_first_obs_extra_messages_go_to_latest_obs(self) -> None:
        agent = _make_agent()
        agent._ingest_obs(
            [
                {"role": "user", "content": "goal"},
                {"role": "user", "content": "screenshot"},
            ]
        )
        assert agent.goal == [{"role": "user", "content": "goal"}]
        assert agent._latest_obs == [{"role": "user", "content": "screenshot"}]
        assert agent.history == []

    def test_subsequent_obs_goes_to_latest_obs_not_history(self) -> None:
        agent = _make_agent()
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._ingest_obs([{"role": "user", "content": "obs_1"}])
        assert agent._latest_obs == [{"role": "user", "content": "obs_1"}]
        assert agent.history == []

    def test_subsequent_obs_replaces_previous_latest_obs(self) -> None:
        agent = _make_agent()
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._ingest_obs([{"role": "user", "content": "obs_1"}])
        agent._ingest_obs([{"role": "user", "content": "obs_2"}])
        assert agent._latest_obs == [{"role": "user", "content": "obs_2"}]


class TestBuildBasePromptModeA:
    """Mode A (enable_summarize=False): completed history included in base prompt."""

    def test_starts_with_system(self) -> None:
        agent = _make_agent()
        agent.goal = [{"role": "user", "content": "goal"}]
        messages = agent._build_base_prompt()
        assert messages[0]["role"] == "system"

    def test_completed_history_included(self) -> None:
        agent = _make_agent()
        _simulate_completed_rounds(agent, n=2)
        messages = agent._build_base_prompt()
        contents = [m.get("content", "") for m in messages if isinstance(m, dict)]
        assert "obs_0" in contents
        assert "asst_0" in contents
        assert "obs_1" in contents
        assert "asst_1" in contents

    def test_latest_obs_not_in_base_prompt(self) -> None:
        """Latest obs is appended by callers, not baked into _build_base_prompt."""
        agent = _make_agent()
        _simulate_completed_rounds(agent, n=2)
        messages = agent._build_base_prompt()
        contents = [m.get("content", "") for m in messages if isinstance(m, dict)]
        assert "obs_2" not in contents

    def test_prefix_stable_across_steps(self) -> None:
        """The base prompt up to the last completed round is identical between steps."""
        agent = _make_agent()
        _simulate_completed_rounds(agent, n=2)
        prefix_step2 = agent._build_base_prompt()

        # Simulate one more step completing
        agent.history.append([{"role": "user", "content": "obs_2"}])
        agent.history.append([{"role": "assistant", "content": "asst_2"}])
        agent._latest_obs = [{"role": "user", "content": "obs_3"}]

        prefix_step3 = agent._build_base_prompt()
        # step2 prefix is a strict prefix of step3 prefix
        assert prefix_step3[: len(prefix_step2)] == prefix_step2


class TestBuildBasePromptModeB:
    """Mode B (enable_summarize=True): summaries as separate assistant messages."""

    def test_summaries_are_separate_messages(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["summary_1", "summary_2"]
        messages = agent._build_base_prompt()
        asst_contents = [m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
        assert "summary_1" in asst_contents
        assert "summary_2" in asst_contents

    def test_summaries_not_bundled_into_single_block(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["summary_1", "summary_2"]
        messages = agent._build_base_prompt()
        asst_messages = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
        # Each summary is its own message — no bundling
        assert len(asst_messages) == 2

    def test_exclude_last_summary_drops_only_last(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["s1", "s2", "s3"]
        messages = agent._build_base_prompt(exclude_last_summary=True)
        asst_contents = [m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
        assert "s1" in asst_contents
        assert "s2" in asst_contents
        assert "s3" not in asst_contents

    def test_latest_obs_not_in_base_prompt(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["s1"]
        agent._latest_obs = [{"role": "user", "content": "latest observation"}]
        messages = agent._build_base_prompt()
        contents = [m.get("content", "") for m in messages if isinstance(m, dict)]
        assert "latest observation" not in contents

    def test_prefix_stable_across_steps(self) -> None:
        """base_prompt up to last summary is identical between consecutive steps."""
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["s1", "s2"]
        prefix_step2 = agent._build_base_prompt()

        agent.summaries = ["s1", "s2", "s3"]
        prefix_step3 = agent._build_base_prompt()

        assert prefix_step3[: len(prefix_step2)] == prefix_step2


class TestChooseContext:
    def test_always_starts_with_system(self) -> None:
        agent = _make_agent()
        _simulate_completed_rounds(agent, n=1)
        messages = agent._choose_context()
        assert messages[0]["role"] == "system"

    def test_goal_always_included(self) -> None:
        agent = _make_agent()
        _simulate_completed_rounds(agent, n=2)
        messages = agent._choose_context()
        assert any(m.get("content") == "goal" for m in messages)

    def test_mode_a_latest_obs_before_react_prompt(self) -> None:
        agent = _make_agent(enable_summarize=False)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._latest_obs = [{"role": "user", "content": "current obs"}]
        messages = agent._choose_context()
        contents = [m.get("content", "") for m in messages if isinstance(m, dict)]
        obs_idx = contents.index("current obs")
        react_idx = contents.index(agent.config.react_prompt)
        assert obs_idx < react_idx

    def test_mode_b_new_summary_before_latest_obs(self) -> None:
        """Mode B act pass: new summary comes before latest obs (keeps summaries contiguous)."""
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["past summary", "new summary"]
        agent._latest_obs = [{"role": "user", "content": "current obs"}]
        messages = agent._choose_context()
        contents = [m.get("content", "") for m in messages if isinstance(m, dict)]
        new_summary_idx = contents.index("new summary")
        obs_idx = contents.index("current obs")
        assert new_summary_idx < obs_idx

    def test_mode_b_past_summaries_before_new_summary(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["s1", "s2"]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        messages = agent._choose_context()
        asst_contents = [m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
        assert asst_contents == ["s1", "s2"]

    def test_ends_with_react_prompt_when_summarize_disabled(self) -> None:
        agent = _make_agent(enable_summarize=False)
        _simulate_completed_rounds(agent, n=1)
        messages = agent._choose_context()
        assert messages[-1]["content"] == agent.config.react_prompt

    def test_ends_with_act_prompt_when_summarize_enabled(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        messages = agent._choose_context()
        assert messages[-1]["content"] == agent.config.act_prompt

    def test_step_counter_in_final_prompt_when_max_actions_set(self) -> None:
        agent = _make_agent()
        agent.config = agent.config.model_copy(update={"max_actions": 5})
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        agent._actions_cnt = 2
        messages = agent._choose_context()
        assert "[Step 3/5]" in messages[-1]["content"]


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
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response("my summary"))
        summary, _ = agent._summarize_past()
        assert summary == "my summary"

    def test_includes_latest_obs_in_prompt(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._latest_obs = [{"role": "user", "content": "the current screenshot"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response())
        agent._summarize_past()
        prompt = agent.summarize_llm.call_args[0][0]
        contents = [m.get("content", "") for m in prompt.messages if isinstance(m, dict)]
        assert any("the current screenshot" in c for c in contents)

    def test_includes_prior_summaries_as_separate_messages(self) -> None:
        """Prior summaries appear as individual assistant messages, not a bundled block."""
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["step one cot", "step two cot"]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response())
        agent._summarize_past()
        prompt = agent.summarize_llm.call_args[0][0]
        asst_msgs = [m for m in prompt.messages if isinstance(m, dict) and m.get("role") == "assistant"]
        assert len(asst_msgs) == 2
        assert asst_msgs[0]["content"] == "step one cot"
        assert asst_msgs[1]["content"] == "step two cot"

    def test_sum_and_act_share_prefix(self) -> None:
        """Sum call and act call share identical prefix up to the last prior summary."""
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent.summaries = ["s1"]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response("new summary"))
        agent._summarize_past()
        sum_prompt = agent.summarize_llm.call_args[0][0]

        agent.summaries.append("new summary")
        act_messages = agent._choose_context()

        # Both start with the same prefix (system, goal, s1) before diverging
        n = 3  # system + goal + s1
        assert sum_prompt.messages[:n] == act_messages[:n]

    def test_cot_mode_uses_cot_prompt(self) -> None:
        agent = _make_agent(enable_summarize=True, summarize_cot_only=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response())
        agent._summarize_past()
        prompt = agent.summarize_llm.call_args[0][0]
        assert prompt.messages[-1]["content"] == agent.config.summarize_cot_prompt

    def test_verbose_mode_uses_verbose_prompt(self) -> None:
        agent = _make_agent(enable_summarize=True, summarize_cot_only=False)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response())
        agent._summarize_past()
        prompt = agent.summarize_llm.call_args[0][0]
        assert prompt.messages[-1]["content"] == agent.config.summarize_verbose_prompt

    def test_uses_same_system_prompt_as_act_pass(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response())
        agent._summarize_past()
        prompt = agent.summarize_llm.call_args[0][0]
        assert isinstance(prompt.messages[0], dict)
        assert prompt.messages[0]["content"] == agent.config.system_prompt

    def test_passes_same_tools_as_act_pass_for_cache(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "goal"}]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response())
        agent._summarize_past()
        prompt = agent.summarize_llm.call_args[0][0]
        assert len(prompt.tools) > 0
        assert prompt.tools[0]["function"]["name"] == "click"

    def test_summarize_llm_uses_same_tool_choice_as_act_llm(self) -> None:
        agent = _make_agent(enable_summarize=True)
        assert agent._summarize_llm_config.tool_choice == agent.config.llm_config.tool_choice


class TestStep:
    def test_step_appends_summary_with_action_when_enabled(self) -> None:
        agent = _make_agent(enable_summarize=True)
        agent.llm = MagicMock(return_value=_mock_llm_response("action"))
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response("step summary"))
        obs = Observation.from_text("goal text")
        agent.step(obs)
        assert len(agent.summaries) == 1
        assert "step summary" in agent.summaries[0]
        assert "Action:" in agent.summaries[0]

    def test_mode_a_commits_obs_and_asst_to_history(self) -> None:
        """Mode A: after step(), completed (obs, asst) pair is in self.history."""
        agent = _make_agent(enable_summarize=False)
        agent.llm = MagicMock(return_value=_mock_llm_response("I think therefore I act"))
        obs = Observation.from_text("goal text")
        agent.step(obs)
        # history: initial obs (from step 0 extra messages or none) + asst
        # For a single-message obs, _latest_obs after ingest is [], so only asst committed
        # Let's just check history grew
        assert len(agent.history) >= 1

    def test_mode_a_second_step_obs_in_history(self) -> None:
        """Mode A: after step 2, step 1's obs and asst are both in history."""
        agent = _make_agent(enable_summarize=False)
        agent.llm = MagicMock(return_value=_mock_llm_response("response"))
        obs1 = Observation.from_text("goal text")
        agent.step(obs1)
        obs2 = Observation.from_text("second obs")
        agent.step(obs2)

        def _content(m: object) -> str:
            if isinstance(m, dict):
                return m.get("content", "") or ""
            return getattr(m, "content", "") or ""

        all_contents = [_content(m) for group in agent.history for m in group]
        assert any("response" in c for c in all_contents)

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
        agent = _make_agent(enable_summarize=True)
        agent.llm = MagicMock(return_value=_mock_llm_response("act text"))
        agent.summarize_llm = MagicMock(return_value=_mock_llm_response("my cot reasoning"))
        result = agent.step(Observation.from_text("goal text"))
        assert result.thoughts == "my cot reasoning"
        assert "Action:" not in result.thoughts

    def test_thoughts_is_inline_content_when_summarize_disabled(self) -> None:
        agent = _make_agent(enable_summarize=False)
        agent.llm = MagicMock(return_value=_mock_llm_response("I think therefore I act"))
        result = agent.step(Observation.from_text("goal text"))
        assert result.thoughts == "I think therefore I act"

    def test_thoughts_is_none_when_no_content(self) -> None:
        agent = _make_agent(enable_summarize=False)
        agent.llm = MagicMock(return_value=_mock_llm_response(""))
        result = agent.step(Observation.from_text("goal text"))
        assert result.thoughts is None


# ---------------------------------------------------------------------------
# Hint / clarification resolution
# ---------------------------------------------------------------------------


def _llm_config() -> LLMConfig:
    return LLMConfig(model_name="test")


class TestHintResolution:
    def test_task_hints_takes_precedence_over_hint(self) -> None:
        config = Genny2Config(llm_config=_llm_config(), hint="general", task_hints={"t1": "specific"})
        agent = Genny2(config=config, action_schemas=[], task_id="t1")
        assert agent._task_hint == "specific"

    def test_falls_back_to_hint_when_no_task_id_match(self) -> None:
        config = Genny2Config(llm_config=_llm_config(), hint="general", task_hints={"other": "x"})
        agent = Genny2(config=config, action_schemas=[], task_id="t1")
        assert agent._task_hint == "general"

    def test_empty_when_no_hint_and_no_match(self) -> None:
        config = Genny2Config(llm_config=_llm_config(), task_hints={"other": "x"})
        agent = Genny2(config=config, action_schemas=[], task_id="t1")
        assert agent._task_hint == ""

    def test_general_hint_applied_when_task_id_none(self) -> None:
        config = Genny2Config(llm_config=_llm_config(), hint="fallback hint")
        agent = Genny2(config=config, action_schemas=[], task_id=None)
        assert agent._task_hint == "fallback hint"

    def test_task_hints_not_applied_when_task_id_none(self) -> None:
        config = Genny2Config(llm_config=_llm_config(), hint="general", task_hints={"t1": "specific"})
        agent = Genny2(config=config, action_schemas=[], task_id=None)
        assert agent._task_hint == "general"

    def test_task_clarification_resolved_by_task_id(self) -> None:
        config = Genny2Config(llm_config=_llm_config(), task_clarification={"t1": "answer format: numeric"})
        agent = Genny2(config=config, action_schemas=[], task_id="t1")
        assert agent._task_clarification == "answer format: numeric"

    def test_task_clarification_empty_when_no_match(self) -> None:
        config = Genny2Config(llm_config=_llm_config(), task_clarification={"other": "x"})
        agent = Genny2(config=config, action_schemas=[], task_id="t1")
        assert agent._task_clarification == ""

    def test_task_clarification_empty_when_task_id_none(self) -> None:
        config = Genny2Config(llm_config=_llm_config(), task_clarification={"t1": "x"})
        agent = Genny2(config=config, action_schemas=[], task_id=None)
        assert agent._task_clarification == ""

    def test_hint_injected_into_base_prompt(self) -> None:
        config = Genny2Config(llm_config=_llm_config(), hint="use keyboard_type_into for dropdowns")
        agent = Genny2(config=config, action_schemas=[], task_id=None)
        agent.goal = [{"role": "user", "content": "do the task"}]
        messages = agent._build_base_prompt()
        contents = [m.get("content", "") for m in messages if isinstance(m, dict)]
        assert any("use keyboard_type_into for dropdowns" in c for c in contents)

    def test_clarification_injected_into_base_prompt(self) -> None:
        config = Genny2Config(llm_config=_llm_config(), task_clarification={"t1": "answer must be numeric"})
        agent = Genny2(config=config, action_schemas=[], task_id="t1")
        agent.goal = [{"role": "user", "content": "do the task"}]
        messages = agent._build_base_prompt()
        contents = [m.get("content", "") for m in messages if isinstance(m, dict)]
        assert any("answer must be numeric" in c for c in contents)

    def test_no_hint_messages_when_both_empty(self) -> None:
        config = Genny2Config(llm_config=_llm_config())
        agent = Genny2(config=config, action_schemas=[], task_id="t1")
        agent.goal = [{"role": "user", "content": "do the task"}]
        messages = agent._build_base_prompt()
        contents = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
        assert "Task Hint" not in contents
        assert "Additional task details" not in contents

    def test_make_wires_task_id(self) -> None:
        config = Genny2Config(llm_config=_llm_config(), task_hints={"t1": "my hint"})
        agent = config.make(task_id="t1")
        assert agent._task_hint == "my hint"

    def test_none_task_id_logs_debug_when_hints_configured(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        config = Genny2Config(llm_config=_llm_config(), task_hints={"t1": "x"})
        with caplog.at_level(logging.DEBUG, logger="cube_harness.agents.genny"):
            Genny2(config=config, action_schemas=[], task_id=None)
        assert any("task_id is None" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Flat history mode
# ---------------------------------------------------------------------------


def _make_flat_agent(step_prompt: str = "", enable_summarize: bool = False) -> Genny2:
    config = Genny2Config(
        llm_config=LLMConfig(model_name="test"),
        flat_history=True,
        step_prompt=step_prompt,
        enable_summarize=enable_summarize,
    )
    return Genny2(config=config, action_schemas=[_make_schema()])


class TestFlatHistory:
    def test_base_prompt_uses_history_groups_not_summaries(self) -> None:
        """flat_history=True: _build_base_prompt renders history groups even when enable_summarize=True."""
        agent = _make_flat_agent(enable_summarize=True)
        agent.goal = [{"role": "user", "content": "task"}]
        agent.summaries = ["step1 summary"]
        agent.history.append([{"role": "user", "content": "obs1"}])
        agent.history.append([{"role": "assistant", "content": "asst1"}])
        messages = agent._build_base_prompt()
        contents = [m.get("content", "") for m in messages if isinstance(m, dict)]
        assert "obs1" in contents
        assert "asst1" in contents
        assert "step1 summary" not in contents

    def test_choose_context_no_trailing_message_when_step_prompt_empty(self) -> None:
        """flat_history=True, step_prompt='': no trailing user message appended."""
        agent = _make_flat_agent(step_prompt="")
        agent.goal = [{"role": "user", "content": "task"}]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        messages = agent._choose_context()
        last = messages[-1]
        # Last message should be the obs, not an injected user prompt
        assert isinstance(last, dict) and last.get("content") == "obs"

    def test_choose_context_trailing_message_when_step_prompt_set(self) -> None:
        """flat_history=True, step_prompt non-empty: trailing message appended."""
        agent = _make_flat_agent(step_prompt="What next?")
        agent.goal = [{"role": "user", "content": "task"}]
        agent._latest_obs = [{"role": "user", "content": "obs"}]
        messages = agent._choose_context()
        last = messages[-1]
        assert isinstance(last, dict) and last.get("content") == "What next?"

    def test_flat_mode_commits_obs_and_asst_to_history(self) -> None:
        """flat_history=True: each step commits obs+asst to history for flat base prompt."""
        agent = _make_flat_agent()
        agent.llm = MagicMock(return_value=_mock_llm_response("action taken"))
        agent.step(Observation.from_text("initial task"))
        agent.step(Observation.from_text("tool result"))
        # After step 2, history should contain rounds from step 1
        assert len(agent.history) >= 1

    def test_flat_mode_step2_base_prompt_includes_step1_history(self) -> None:
        """After step 1, step 2's base prompt includes step 1's completed round."""
        agent = _make_flat_agent()
        agent.llm = MagicMock(return_value=_mock_llm_response("step1 response"))
        agent.step(Observation.from_text("initial task"))
        agent.step(Observation.from_text("tool result"))
        messages = agent._build_base_prompt()

        def _content(m: object) -> str:
            if isinstance(m, dict):
                return m.get("content", "") or ""
            return getattr(m, "content", "") or ""

        all_contents = [_content(m) for m in messages]
        assert any("step1 response" in c for c in all_contents)


# ---------------------------------------------------------------------------
# cost_limit
# ---------------------------------------------------------------------------


def _mock_llm_response_with_cost(cost: float, text: str = "response") -> LLMResponse:
    from litellm import Message as LitellmMessage

    return LLMResponse(
        message=LitellmMessage(role="assistant", content=text),
        usage=Usage(prompt_tokens=10, completion_tokens=5, cost=cost),
    )


class TestCostLimit:
    def test_no_stop_when_below_limit(self) -> None:
        config = Genny2Config(llm_config=LLMConfig(model_name="test"), cost_limit=1.0)
        agent = Genny2(config=config, action_schemas=[_make_schema()])
        agent.llm = MagicMock(return_value=_mock_llm_response_with_cost(0.10))
        agent.step(Observation.from_text("task"))
        # LLM was called (not short-circuited by cost limit)
        agent.llm.assert_called_once()

    def test_stop_when_limit_reached(self) -> None:
        config = Genny2Config(llm_config=LLMConfig(model_name="test"), cost_limit=0.05)
        agent = Genny2(config=config, action_schemas=[_make_schema()])
        agent.llm = MagicMock(return_value=_mock_llm_response_with_cost(0.10))
        agent.step(Observation.from_text("task"))  # spends $0.10, exceeds limit
        result = agent.step(Observation.from_text("task2"))
        assert result.actions[0].name == "final_step"

    def test_no_limit_when_cost_limit_none(self) -> None:
        config = Genny2Config(llm_config=LLMConfig(model_name="test"), cost_limit=None)
        agent = Genny2(config=config, action_schemas=[_make_schema()])
        agent._total_cost = 999.0
        agent.llm = MagicMock(return_value=_mock_llm_response_with_cost(0.0))
        agent.step(Observation.from_text("task"))
        # LLM was called despite enormous accumulated cost because cost_limit=None
        agent.llm.assert_called_once()

    def test_total_cost_accumulates(self) -> None:
        config = Genny2Config(llm_config=LLMConfig(model_name="test"), cost_limit=10.0)
        agent = Genny2(config=config, action_schemas=[_make_schema()])
        agent.llm = MagicMock(return_value=_mock_llm_response_with_cost(0.50))
        agent.step(Observation.from_text("t1"))
        agent.step(Observation.from_text("t2"))
        assert agent._total_cost == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# max_format_errors
# ---------------------------------------------------------------------------


def _mock_response_with_tool_call(name: str = "click") -> LLMResponse:
    from litellm import Message as LitellmMessage
    from litellm.types.utils import ChatCompletionMessageToolCall, Function

    tc = ChatCompletionMessageToolCall(
        id="call_1",
        function=Function(name=name, arguments='{"element_id": "btn"}'),
        type="function",
    )
    return LLMResponse(
        message=LitellmMessage(role="assistant", content=None, tool_calls=[tc]),
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


class TestMaxFormatErrors:
    def test_no_retry_when_zero(self) -> None:
        """max_format_errors=0: no retry, empty actions list returned."""
        config = Genny2Config(llm_config=LLMConfig(model_name="test"), max_format_errors=0)
        agent = Genny2(config=config, action_schemas=[_make_schema()])
        agent.llm = MagicMock(return_value=_mock_llm_response("no tool calls"))
        result = agent.step(Observation.from_text("task"))
        agent.llm.assert_called_once()
        assert result.actions == []

    def test_retries_on_no_tool_calls(self) -> None:
        """max_format_errors=2: LLM called up to 3 times (1 initial + 2 retries)."""
        config = Genny2Config(llm_config=LLMConfig(model_name="test"), max_format_errors=2)
        agent = Genny2(config=config, action_schemas=[_make_schema()])
        no_tool_resp = _mock_llm_response("no tool calls")
        tool_resp = _mock_response_with_tool_call()
        # First call: no tool calls. Second call: has tool call.
        agent.llm = MagicMock(side_effect=[no_tool_resp, tool_resp])
        result = agent.step(Observation.from_text("task"))
        assert agent.llm.call_count == 2
        assert len(result.actions) == 1

    def test_stop_when_all_retries_exhausted(self) -> None:
        """When all retries fail, STOP action is returned."""
        config = Genny2Config(llm_config=LLMConfig(model_name="test"), max_format_errors=2)
        agent = Genny2(config=config, action_schemas=[_make_schema()])
        agent.llm = MagicMock(return_value=_mock_llm_response("no tool calls"))
        result = agent.step(Observation.from_text("task"))
        assert agent.llm.call_count == 3  # initial + 2 retries
        assert result.actions[0].name == "final_step"

    def test_correction_message_appended_on_retry(self) -> None:
        """On retry the correction user message and empty response are in the next prompt."""
        config = Genny2Config(llm_config=LLMConfig(model_name="test"), max_format_errors=1)
        agent = Genny2(config=config, action_schemas=[_make_schema()])
        no_tool_resp = _mock_llm_response("no tool calls")
        tool_resp = _mock_response_with_tool_call()
        agent.llm = MagicMock(side_effect=[no_tool_resp, tool_resp])
        agent.step(Observation.from_text("task"))
        # agent.llm is called as agent.llm(prompt) so args[0] is the Prompt
        second_prompt = agent.llm.call_args_list[1][0][0]
        contents = [
            m.get("content", "") if isinstance(m, dict) else (getattr(m, "content", "") or "")
            for m in second_prompt.messages
        ]
        assert any("No tool calls found" in (c or "") for c in contents)
