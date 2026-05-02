"""Tests for cube_harness.llm module."""

import json
from unittest.mock import MagicMock, patch

from litellm import Message
from litellm.types.utils import ChatCompletionMessageToolCall, Function

from cube_harness.llm import (
    LLM,
    LLMCall,
    LLMConfig,
    Prompt,
    _build_cache_injection_points,
    _is_anthropic_model,
    _mark_last_tool_for_cache,
)


class TestPrompt:
    """Tests for Prompt class."""

    def test_prompt_creation(self, sample_prompt):
        """Test basic Prompt creation."""
        assert len(sample_prompt.messages) == 2
        assert sample_prompt.tools == []

    def test_prompt_with_tools(self):
        """Test Prompt with tools."""
        tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        prompt = Prompt(messages=[{"role": "user", "content": "Search for X"}], tools=tools)
        assert len(prompt.tools) == 1
        assert prompt.tools[0]["function"]["name"] == "search"

    def test_prompt_empty_defaults(self):
        """Test Prompt with minimal arguments."""
        prompt = Prompt(messages=[])
        assert prompt.messages == []
        assert prompt.tools == []

    def test_prompt_str_representation(self, sample_prompt):
        """Test Prompt string representation."""
        str_repr = str(sample_prompt)
        assert "Messages[2]" in str_repr
        assert "system" in str_repr
        assert "user" in str_repr

    def test_prompt_serialization(self, sample_prompt):
        """Test Prompt JSON serialization."""
        json_str = sample_prompt.model_dump_json()
        data = json.loads(json_str)
        assert len(data["messages"]) == 2
        assert data["tools"] == []


class TestLLMConfig:
    """Tests for LLMConfig class."""

    def test_llm_config_creation(self, sample_llm_config):
        """Test basic LLMConfig creation."""
        assert sample_llm_config.model_name == "gpt-5-nano"
        assert sample_llm_config.temperature == 0.7
        assert sample_llm_config.max_tokens == 4096

    def test_llm_config_defaults(self):
        """Test LLMConfig with default values."""
        config = LLMConfig(model_name="gpt-4")
        assert config.temperature == 1.0
        assert config.max_tokens == 128000
        assert config.max_completion_tokens == 8192
        assert config.reasoning_effort is None
        assert config.tool_choice == "auto"
        assert config.parallel_tool_calls is False
        assert config.num_retries == 5
        assert config.retry_strategy == "exponential_backoff_retry"

    def test_llm_config_custom_values(self):
        """Test LLMConfig with custom values."""
        config = LLMConfig(
            model_name="claude-3-opus",
            temperature=0.5,
            max_tokens=10000,
            max_completion_tokens=2048,
            reasoning_effort="high",
            tool_choice="required",
            parallel_tool_calls=True,
            num_retries=3,
            retry_strategy="constant_retry",
        )
        assert config.reasoning_effort == "high"
        assert config.tool_choice == "required"
        assert config.parallel_tool_calls is True
        assert config.retry_strategy == "constant_retry"

    def test_llm_config_make(self, sample_llm_config):
        """Test creating LLM instance from config."""
        llm = sample_llm_config.make()
        assert isinstance(llm, LLM)
        assert llm.config == sample_llm_config

    def test_llm_config_make_counter(self, sample_llm_config):
        """Test creating token counter from config."""
        counter = sample_llm_config.make_counter()
        assert callable(counter)

    def test_llm_config_serialization(self, sample_llm_config):
        """Test LLMConfig JSON serialization."""
        json_str = sample_llm_config.model_dump_json()
        data = json.loads(json_str)
        assert data["model_name"] == "gpt-5-nano"


class TestLLM:
    """Tests for LLM class."""

    def test_llm_initialization(self, sample_llm_config):
        """Test LLM initialization."""
        llm = LLM(config=sample_llm_config)
        assert llm.config == sample_llm_config

    @patch("cube_harness.llm.completion_with_retries")
    def test_llm_call(self, mock_completion, sample_llm_config, sample_prompt):
        """Test LLM call with mocked completion."""
        mock_message = Message(role="assistant", content="Hello! How can I help?")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_response.usage = None  # Simulate no usage info
        mock_completion.return_value = mock_response

        llm = LLM(config=sample_llm_config)
        result = llm(sample_prompt)

        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5-nano"
        assert call_kwargs["temperature"] == 0.7
        assert result.message.content == "Hello! How can I help?"
        assert result.usage.prompt_tokens == 0  # No usage info provided

    @patch("cube_harness.llm.completion_with_retries")
    def test_llm_call_with_tools(self, mock_completion, sample_llm_config) -> None:
        """Test LLM call with tools."""
        tool_call = ChatCompletionMessageToolCall(
            id="call_1", function=Function(name="search", arguments='{"query": "test"}'), type="function"
        )
        mock_message = Message(role="assistant", content="Tool call made.", tool_calls=[tool_call])
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_response.usage = None  # Simulate no usage info
        mock_completion.return_value = mock_response

        tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        prompt = Prompt(messages=[{"role": "user", "content": "Search"}], tools=tools)

        llm = LLM(config=sample_llm_config)
        result = llm(prompt)

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["tools"] == tools
        assert result.message.tool_calls is not None
        assert result.message.content == "Tool call made."


class TestLLMCall:
    """Tests for LLMCall class."""

    def test_llm_call_creation(self, sample_llm_config, sample_prompt):
        """Test basic LLMCall creation."""
        mock_message = Message(role="assistant", content="Test response")
        llm_call = LLMCall(llm_config=sample_llm_config, prompt=sample_prompt, output=mock_message)

        assert llm_call.llm_config == sample_llm_config
        assert llm_call.prompt == sample_prompt
        assert llm_call.output == mock_message
        assert llm_call.id is not None
        assert llm_call.timestamp is not None

    def test_llm_call_auto_id(self, sample_llm_config, sample_prompt):
        """Test LLMCall auto-generates unique IDs."""
        mock_message = Message(role="assistant", content="Test")
        call1 = LLMCall(llm_config=sample_llm_config, prompt=sample_prompt, output=mock_message)
        call2 = LLMCall(llm_config=sample_llm_config, prompt=sample_prompt, output=mock_message)
        assert call1.id != call2.id

    def test_llm_call_timestamp(self, sample_llm_config, sample_prompt):
        """Test LLMCall has timestamp."""
        mock_message = Message(role="assistant", content="Test")
        llm_call = LLMCall(llm_config=sample_llm_config, prompt=sample_prompt, output=mock_message)
        # Check timestamp is ISO format
        assert "T" in llm_call.timestamp

    def test_llm_call_serialization(self, sample_llm_config, sample_prompt):
        """Test LLMCall JSON serialization."""
        mock_message = Message(role="assistant", content="Test response")
        llm_call = LLMCall(llm_config=sample_llm_config, prompt=sample_prompt, output=mock_message)
        json_str = llm_call.model_dump_json()
        data = json.loads(json_str)
        assert "id" in data
        assert "timestamp" in data
        assert data["llm_config"]["model_name"] == "gpt-5-nano"


class TestCacheControlHelpers:
    """Tests for prompt-cache helper functions."""

    def test_is_anthropic_model_true_cases(self) -> None:
        for name in [
            "claude-3-5-sonnet-20241022",
            "anthropic/claude-3-5-haiku",
            "bedrock/anthropic.claude-sonnet-4-5",
            "vertex_ai/claude-opus-4-1",
        ]:
            assert _is_anthropic_model(name), name

    def test_is_anthropic_model_false_cases(self) -> None:
        for name in ["gpt-4o", "gpt-5-mini", "azure/gpt-4o", "gemini/gemini-2.0-flash"]:
            assert not _is_anthropic_model(name), name

    def test_injection_points_empty_messages(self) -> None:
        assert _build_cache_injection_points([]) == []

    def test_injection_points_system_only(self) -> None:
        points = _build_cache_injection_points([{"role": "system", "content": "x"}])
        # No assistant message; only system anchor.
        assert points == [{"location": "message", "index": 0, "control": {"type": "ephemeral"}}]

    def test_injection_points_system_user_assistant(self) -> None:
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
        points = _build_cache_injection_points(msgs)
        assert points == [
            {"location": "message", "index": 0, "control": {"type": "ephemeral"}},
            {"location": "message", "index": 2, "control": {"type": "ephemeral"}},
        ]

    def test_injection_points_uses_last_assistant(self) -> None:
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
        ]
        points = _build_cache_injection_points(msgs)
        indices = sorted(p["index"] for p in points)
        assert indices == [0, 4]

    def test_injection_points_no_system(self) -> None:
        msgs = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u2"},
        ]
        points = _build_cache_injection_points(msgs)
        # No system message → only the rolling assistant breakpoint.
        assert points == [{"location": "message", "index": 1, "control": {"type": "ephemeral"}}]

    def test_injection_points_message_objects(self) -> None:
        msgs = [Message(role="system", content="s"), Message(role="assistant", content="a")]
        points = _build_cache_injection_points(msgs)
        indices = sorted(p["index"] for p in points)
        assert indices == [0, 1]

    def test_mark_last_tool_for_cache_empty(self) -> None:
        assert _mark_last_tool_for_cache([]) == []

    def test_mark_last_tool_for_cache_marks_last(self) -> None:
        tools = [
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}},
        ]
        marked = _mark_last_tool_for_cache(tools)
        assert marked[0] == tools[0]  # first untouched
        assert marked[1]["cache_control"] == {"type": "ephemeral"}
        # Original list not mutated.
        assert "cache_control" not in tools[1]


class TestLLMCacheControl:
    """Tests for LLM.__call__ cache_control wiring."""

    @patch("cube_harness.llm.completion_with_retries")
    def test_no_cache_control_when_unset(self, mock_completion, sample_prompt) -> None:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=Message(role="assistant", content="x"))], usage=None
        )
        llm = LLM(config=LLMConfig(model_name="claude-sonnet-4-5"))  # set_cache_control=None
        llm(sample_prompt)
        kwargs = mock_completion.call_args.kwargs
        assert "cache_control_injection_points" not in kwargs
        for tool in kwargs.get("tools", []):
            assert "cache_control" not in tool

    @patch("cube_harness.llm.completion_with_retries")
    def test_no_cache_control_for_non_anthropic(self, mock_completion) -> None:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=Message(role="assistant", content="x"))], usage=None
        )
        llm = LLM(config=LLMConfig(model_name="gpt-4o", set_cache_control="auto"))
        prompt = Prompt(
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            tools=[{"type": "function", "function": {"name": "f"}}],
        )
        llm(prompt)
        kwargs = mock_completion.call_args.kwargs
        # Non-Anthropic: don't inject cache_control even when set_cache_control="auto".
        assert "cache_control_injection_points" not in kwargs
        for tool in kwargs.get("tools", []):
            assert "cache_control" not in tool

    @patch("cube_harness.llm.completion_with_retries")
    def test_auto_caching_for_anthropic(self, mock_completion) -> None:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=Message(role="assistant", content="x"))], usage=None
        )
        llm = LLM(config=LLMConfig(model_name="claude-sonnet-4-5", set_cache_control="auto"))
        prompt = Prompt(
            messages=[
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "u2"},
            ],
            tools=[
                {"type": "function", "function": {"name": "a"}},
                {"type": "function", "function": {"name": "b"}},
            ],
        )
        llm(prompt)
        kwargs = mock_completion.call_args.kwargs
        points = kwargs["cache_control_injection_points"]
        indices = sorted(p["index"] for p in points)
        assert indices == [0, 2]
        # Last tool marked, first untouched. Original prompt.tools must not be mutated.
        assert kwargs["tools"][1]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in kwargs["tools"][0]
        assert "cache_control" not in prompt.tools[1]

    @patch("cube_harness.llm.completion_with_retries")
    def test_auto_caching_no_assistant_yet(self, mock_completion) -> None:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=Message(role="assistant", content="x"))], usage=None
        )
        llm = LLM(config=LLMConfig(model_name="claude-sonnet-4-5", set_cache_control="auto"))
        prompt = Prompt(
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            tools=[],
        )
        llm(prompt)
        kwargs = mock_completion.call_args.kwargs
        # Only system-message anchor when no assistant has spoken yet.
        assert kwargs["cache_control_injection_points"] == [
            {"location": "message", "index": 0, "control": {"type": "ephemeral"}}
        ]
