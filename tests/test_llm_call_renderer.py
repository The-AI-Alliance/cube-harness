"""Tests for agentlab2.rl.llm_call_renderer module."""

import pytest
from litellm import Message

from agentlab2.llm import LLMCall, LLMConfig, Prompt
from agentlab2.rl.llm_call_renderer import (
    _load_apriel_tokenizer,
    _render_apriel_output,
    llm_call_to_text_pair,
)

_APRIEL_CONFIG = LLMConfig(model_name="hosted_vllm/ServiceNow-AI/Apriel-1.6-15b-Thinker")


def _make_tool_call(name: str, arguments: str, tc_id: str = "call_1") -> dict:
    """Helper to build a tool call dict."""
    return {
        "id": tc_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _make_llm_call(
    messages: list[dict],
    output: Message,
    tools: list[dict] | None = None,
) -> LLMCall:
    """Helper to build an LLMCall."""
    return LLMCall(
        llm_config=_APRIEL_CONFIG,
        prompt=Prompt(messages=messages, tools=tools or []),
        output=output,
    )


# ---------------------------------------------------------------------------
# Unit tests for _render_apriel_output (no tokenizer / network needed)
# ---------------------------------------------------------------------------


class TestRenderAprielOutput:
    """Unit tests for _render_apriel_output."""

    def test_content_only(self) -> None:
        output = Message(content="Hello world.", role="assistant")
        result = _render_apriel_output(output)
        assert result == "Hello world.\n<|end|>\n"

    def test_tool_calls_only(self) -> None:
        output = Message(
            content=None,
            role="assistant",
            tool_calls=[_make_tool_call("browser_click", '{"bid": "btn"}')],
        )
        result = _render_apriel_output(output)
        assert (
            result == '\n<tool_calls>[{"name": "browser_click", "arguments": {"bid": "btn"}}]</tool_calls>\n<|end|>\n'
        )

    def test_content_and_tool_calls(self) -> None:
        output = Message(
            content="Let me click.",
            role="assistant",
            tool_calls=[_make_tool_call("browser_click", '{"bid": "btn"}')],
        )
        result = _render_apriel_output(output)
        assert result.startswith("Let me click.")
        assert "<tool_calls>" in result
        assert result.endswith("\n<|end|>\n")

    def test_multiple_tool_calls(self) -> None:
        output = Message(
            content=None,
            role="assistant",
            tool_calls=[
                _make_tool_call("browser_type", '{"bid": "input", "text": "hello"}', tc_id="c1"),
                _make_tool_call("browser_click", '{"bid": "submit"}', tc_id="c2"),
            ],
        )
        result = _render_apriel_output(output)
        assert "browser_type" in result
        assert "browser_click" in result
        assert result.count('"name"') == 2

    def test_no_tool_ids_in_output(self) -> None:
        output = Message(
            content=None,
            role="assistant",
            tool_calls=[_make_tool_call("noop", "{}", tc_id="call_xyz")],
        )
        result = _render_apriel_output(output)
        assert "call_xyz" not in result
        assert '"id"' not in result

    def test_ends_with_end_token(self) -> None:
        output = Message(content="test", role="assistant")
        assert _render_apriel_output(output).endswith("\n<|end|>\n")

    def test_empty_content_treated_as_no_content(self) -> None:
        output = Message(content="", role="assistant")
        result = _render_apriel_output(output)
        assert result == "\n<|end|>\n"

    def test_reasoning_with_tool_calls(self) -> None:
        """Reasoning content should appear before [BEGIN FINAL RESPONSE] marker."""
        output = Message(
            content=None,
            role="assistant",
            reasoning_content="I need to call get_weather for Montreal.",
            tool_calls=[_make_tool_call("get_weather", '{"city": "Montreal"}')],
        )
        result = _render_apriel_output(output)
        assert result == (
            "I need to call get_weather for Montreal."
            "\n[BEGIN FINAL RESPONSE]"
            '\n<tool_calls>[{"name": "get_weather", "arguments": {"city": "Montreal"}}]</tool_calls>'
            "\n<|end|>\n"
        )

    def test_reasoning_with_content(self) -> None:
        """Reasoning + content (no tool calls)."""
        output = Message(
            content="The answer is 42.",
            role="assistant",
            reasoning_content="Let me think step by step.",
        )
        result = _render_apriel_output(output)
        assert result == ("Let me think step by step.\n[BEGIN FINAL RESPONSE]\nThe answer is 42.\n<|end|>\n")

    def test_reasoning_with_content_and_tool_calls(self) -> None:
        """Reasoning + content + tool calls."""
        output = Message(
            content="I'll check the weather.",
            role="assistant",
            reasoning_content="The user wants weather info.",
            tool_calls=[_make_tool_call("get_weather", '{"city": "Montreal"}')],
        )
        result = _render_apriel_output(output)
        assert result.startswith("The user wants weather info.\n[BEGIN FINAL RESPONSE]\nI'll check the weather.")
        assert "<tool_calls>" in result

    def test_no_reasoning_no_marker(self) -> None:
        """Without reasoning_content, no [BEGIN FINAL RESPONSE] marker should appear."""
        output = Message(content="Just a response.", role="assistant")
        result = _render_apriel_output(output)
        assert "[BEGIN FINAL RESPONSE]" not in result

    def test_matches_live_vllm_output(self) -> None:
        """Reproduce the exact output from a real vLLM Apriel response."""
        output = Message(
            content=None,
            role="assistant",
            reasoning_content=(
                'The user asks: "What is the weather in Montreal?" '
                "We need to get current weather for Montreal using the function get_weather. "
                'We\'ll call the function with city: "Montreal".'
            ),
            tool_calls=[_make_tool_call("get_weather", '{"city": "Montreal"}')],
        )
        expected_raw = (
            'The user asks: "What is the weather in Montreal?" '
            "We need to get current weather for Montreal using the function get_weather. "
            'We\'ll call the function with city: "Montreal".'
            "\n[BEGIN FINAL RESPONSE]"
            '\n<tool_calls>[{"name": "get_weather", "arguments": {"city": "Montreal"}}]</tool_calls>'
            "\n<|end|>\n"
        )
        assert _render_apriel_output(output) == expected_raw


# ---------------------------------------------------------------------------
# Dispatching tests
# ---------------------------------------------------------------------------


class TestLlmCallToTextPair:
    """Tests for llm_call_to_text_pair dispatching and text_in structure."""

    def test_unsupported_model_raises(self) -> None:
        llm_call = LLMCall(
            llm_config=LLMConfig(model_name="unknown-model"),
            prompt=Prompt(messages=[{"role": "user", "content": "hi"}]),
            output=Message(content="hello", role="assistant"),
        )
        with pytest.raises(NotImplementedError, match="unknown-model"):
            llm_call_to_text_pair(llm_call, step_reward=0.0)

    @pytest.mark.slow
    def test_text_in_has_special_tokens(self) -> None:
        llm_call = _make_llm_call(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Do something."},
            ],
            output=Message(content="Done.", role="assistant"),
        )
        result = llm_call_to_text_pair(llm_call, step_reward=0.0)
        assert "<|begin_system|>" in result.prompt_text
        assert "<|begin_user|>" in result.prompt_text
        assert "<|begin_assistant|>" in result.prompt_text
        assert "Here are my reasoning steps:" in result.prompt_text

    @pytest.mark.slow
    def test_history_tool_ids_present_in_input_not_output(self) -> None:
        llm_call = _make_llm_call(
            messages=[
                {"role": "user", "content": "Click it."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_make_tool_call("browser_click", '{"bid": "btn"}', tc_id="c_hist")],
                },
                {"role": "tool", "content": "Clicked.", "tool_call_id": "c_hist"},
                {"role": "user", "content": "Now what?"},
            ],
            output=Message(content="Done.", role="assistant"),
        )
        result = llm_call_to_text_pair(llm_call, step_reward=1.0)
        assert '"id": "c_hist"' in result.prompt_text
        assert "c_hist" not in result.response_text
        assert result.reward == 1.0
        assert result.images == []


# ---------------------------------------------------------------------------
# Template conformance: compare _render_apriel_output against
# tokenizer.apply_chat_template to guarantee identical rendering.
#
# Note: these tests validate the content/tool_calls structure. The
# reasoning_content field is a vLLM runtime concept (split from the raw
# output) and is not represented separately in the chat template.
# The live vLLM test (test_matches_live_vllm_output) covers that path.
# ---------------------------------------------------------------------------


class TestOutputMatchesTemplate:
    """Verify _render_apriel_output matches apply_chat_template output exactly."""

    def _assert_matches_template(self, output_msg: Message, output_dict: dict) -> None:
        """Render a full conversation through the tokenizer and compare the
        assistant block with our _render_apriel_output."""
        tokenizer = _load_apriel_tokenizer()

        messages = [{"role": "user", "content": "Do something."}]
        full_messages = messages + [output_dict]
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            training_prompt=True,
        )

        marker = "\n<|begin_assistant|>\n"
        template_output = full_text[full_text.rfind(marker) + len(marker) :]

        our_output = _render_apriel_output(output_msg)
        assert our_output == template_output

    @pytest.mark.slow
    def test_content_only(self) -> None:
        self._assert_matches_template(
            Message(content="I will help you.", role="assistant"),
            {"role": "assistant", "content": "I will help you."},
        )

    @pytest.mark.slow
    def test_tool_calls_only(self) -> None:
        tc = _make_tool_call("browser_click", '{"bid": "btn"}')
        self._assert_matches_template(
            Message(content=None, role="assistant", tool_calls=[tc]),
            {"role": "assistant", "content": None, "tool_calls": [tc]},
        )

    @pytest.mark.slow
    def test_content_and_tool_calls(self) -> None:
        tc = _make_tool_call("browser_click", '{"bid": "btn"}')
        self._assert_matches_template(
            Message(content="Let me click.", role="assistant", tool_calls=[tc]),
            {"role": "assistant", "content": "Let me click.", "tool_calls": [tc]},
        )

    @pytest.mark.slow
    def test_multiple_tool_calls(self) -> None:
        tcs = [
            _make_tool_call("browser_type", '{"bid": "input", "text": "hello"}', tc_id="c1"),
            _make_tool_call("browser_click", '{"bid": "submit"}', tc_id="c2"),
        ]
        self._assert_matches_template(
            Message(content=None, role="assistant", tool_calls=tcs),
            {"role": "assistant", "content": None, "tool_calls": tcs},
        )

    @pytest.mark.slow
    def test_multiline_content(self) -> None:
        content = "Here is the result:\n- Item 1\n- Item 2"
        self._assert_matches_template(
            Message(content=content, role="assistant"),
            {"role": "assistant", "content": content},
        )

    @pytest.mark.slow
    def test_reasoning_roundtrip(self) -> None:
        """Full content (reasoning + marker + response) rendered through the
        template should match our reconstruction from split fields."""
        reasoning = "Step 1: analyze the question."
        final_content = "The answer is 42."
        full_content = f"{reasoning}\n[BEGIN FINAL RESPONSE]\n{final_content}"

        # Template sees the full content as one string
        template_dict = {"role": "assistant", "content": full_content}
        # Our function sees it split (as vLLM returns it)
        output_msg = Message(content=final_content, role="assistant", reasoning_content=reasoning)

        tokenizer = _load_apriel_tokenizer()
        messages = [{"role": "user", "content": "What is the answer?"}]
        full_text = tokenizer.apply_chat_template(
            messages + [template_dict],
            tokenize=False,
            add_generation_prompt=False,
            training_prompt=True,
        )

        marker = "\n<|begin_assistant|>\n"
        template_output = full_text[full_text.rfind(marker) + len(marker) :]

        our_output = _render_apriel_output(output_msg)
        assert our_output == template_output
