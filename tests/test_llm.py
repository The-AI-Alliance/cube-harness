"""Tests for cube_harness.llm module."""

import json
from unittest.mock import MagicMock, patch

import pytest
import tenacity
from litellm import Message
from litellm.types.utils import ChatCompletionMessageToolCall, Function

from cube_harness.llm import LLM, LLMCall, LLMConfig, Prompt


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

    @patch("cube_harness.llm.litellm.completion")
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

    @patch("cube_harness.llm.litellm.completion")
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


class TestRetryBackoff:
    """Tests for LLM._completion_with_retry tenacity-based retry."""

    def _make_ok_response(self) -> MagicMock:
        return MagicMock(
            choices=[MagicMock(message=Message(role="assistant", content="ok"))],
            usage=None,
        )

    @patch("cube_harness.llm.litellm.completion")
    def test_succeeds_on_first_attempt(self, mock_completion: MagicMock) -> None:
        """No retry needed when the first call succeeds."""
        mock_completion.return_value = self._make_ok_response()
        llm = LLM(config=LLMConfig(model_name="gpt-5-nano", num_retries=3))
        prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[])
        llm(prompt)
        assert mock_completion.call_count == 1

    @patch("cube_harness.llm.litellm.completion")
    def test_retries_on_transient_error_then_succeeds(self, mock_completion: MagicMock) -> None:
        """Retries when first call raises a retriable error, then succeeds."""
        from litellm.exceptions import InternalServerError

        mock_completion.side_effect = [
            InternalServerError("overloaded", llm_provider="anthropic", model="claude"),
            self._make_ok_response(),
        ]
        llm = LLM(config=LLMConfig(model_name="gpt-5-nano", num_retries=3))
        prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[])
        with patch("tenacity.wait_exponential", return_value=tenacity.wait_none()):
            result = llm(prompt)
        assert mock_completion.call_count == 2
        assert result.message.content == "ok"

    @patch("cube_harness.llm.litellm.completion")
    def test_does_not_retry_permanent_errors(self, mock_completion: MagicMock) -> None:
        """Non-retriable errors (e.g. NotFoundError) are raised immediately without retry."""
        from litellm.exceptions import NotFoundError

        mock_completion.side_effect = NotFoundError("model not found", llm_provider="openai", model="bad-model")
        llm = LLM(config=LLMConfig(model_name="bad-model", num_retries=3))
        prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[])
        with pytest.raises(NotFoundError):
            llm(prompt)
        assert mock_completion.call_count == 1

    @patch("cube_harness.llm.litellm.completion")
    def test_reraises_after_exhausting_retries(self, mock_completion: MagicMock) -> None:
        """After num_retries attempts all failing, the exception is re-raised."""
        from litellm.exceptions import RateLimitError

        mock_completion.side_effect = RateLimitError("rate limited", llm_provider="openai", model="gpt-5-nano")
        llm = LLM(config=LLMConfig(model_name="gpt-5-nano", num_retries=2))
        prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[])
        with patch("tenacity.wait_exponential", return_value=tenacity.wait_none()):
            with pytest.raises(RateLimitError):
                llm(prompt)
        assert mock_completion.call_count == 2


class TestRetryIntegration:
    """Integration tests for LLM retry — tenacity loop runs for real, only the
    underlying litellm.completion call is stubbed.  We patch time.sleep to keep
    tests fast without bypassing tenacity's own retry/stop/wait machinery."""

    def _ok_response(self) -> MagicMock:
        return MagicMock(
            choices=[MagicMock(message=Message(role="assistant", content="ok"))],
            usage=None,
        )

    @patch("time.sleep")
    @patch("cube_harness.llm.litellm.completion")
    def test_full_retry_loop_attempt_count(self, mock_completion: MagicMock, mock_sleep: MagicMock) -> None:
        """Tenacity calls litellm exactly num_retries times when every attempt fails."""
        from litellm.exceptions import ServiceUnavailableError

        mock_completion.side_effect = ServiceUnavailableError("unavailable", llm_provider="anthropic", model="claude")
        llm = LLM(config=LLMConfig(model_name="claude-sonnet", num_retries=3))
        prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[])
        with pytest.raises(ServiceUnavailableError):
            llm(prompt)
        assert mock_completion.call_count == 3
        # tenacity slept between attempts (num_retries-1 sleeps for num_retries attempts)
        assert mock_sleep.call_count == 2

    @patch("time.sleep")
    @patch("cube_harness.llm.litellm.completion")
    def test_sleep_duration_is_exponential(self, mock_completion: MagicMock, mock_sleep: MagicMock) -> None:
        """Sleep durations grow exponentially (multiplier=2): ~2s, ~4s."""
        from litellm.exceptions import InternalServerError

        mock_completion.side_effect = [
            InternalServerError("err", llm_provider="anthropic", model="claude"),
            InternalServerError("err", llm_provider="anthropic", model="claude"),
            self._ok_response(),
        ]
        llm = LLM(config=LLMConfig(model_name="claude-sonnet", num_retries=3))
        prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[])
        llm(prompt)
        assert mock_completion.call_count == 3
        assert mock_sleep.call_count == 2
        first_sleep, second_sleep = (c.args[0] for c in mock_sleep.call_args_list)
        # multiplier=2: wait = 2 * 2^(attempt-1), capped at 120
        assert 1.9 <= first_sleep <= 2.1, f"expected ~2s, got {first_sleep}"
        assert 3.9 <= second_sleep <= 4.1, f"expected ~4s, got {second_sleep}"

    @patch("time.sleep")
    @patch("cube_harness.llm.litellm.completion")
    def test_num_retries_zero_makes_single_attempt(self, mock_completion: MagicMock, mock_sleep: MagicMock) -> None:
        """num_retries=0 makes exactly one attempt with no sleep — never zero attempts."""
        from litellm.exceptions import RateLimitError

        mock_completion.side_effect = RateLimitError("limited", llm_provider="openai", model="gpt")
        llm = LLM(config=LLMConfig(model_name="gpt-5-nano", num_retries=0))
        prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[])
        with pytest.raises(RateLimitError):
            llm(prompt)
        # Exactly 1 attempt — tenacity's minimum even with stop_after_attempt(0)
        assert mock_completion.call_count == 1
        assert mock_sleep.call_count == 0

    @patch("time.sleep")
    @patch("cube_harness.llm.litellm.completion")
    def test_permanent_error_no_sleep(self, mock_completion: MagicMock, mock_sleep: MagicMock) -> None:
        """A non-retriable error raises immediately — tenacity never sleeps."""
        from litellm.exceptions import NotFoundError

        mock_completion.side_effect = NotFoundError("model not found", llm_provider="openai", model="bad")
        llm = LLM(config=LLMConfig(model_name="bad", num_retries=5))
        prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[])
        with pytest.raises(NotFoundError):
            llm(prompt)
        assert mock_completion.call_count == 1
        assert mock_sleep.call_count == 0

    @patch("time.sleep")
    @patch("cube_harness.llm.litellm.completion")
    def test_success_after_retries_returns_correct_response(
        self, mock_completion: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """The response from the successful attempt is returned, not a retry artefact."""
        from litellm.exceptions import Timeout

        success = MagicMock(
            choices=[MagicMock(message=Message(role="assistant", content="final answer"))],
            usage=None,
        )
        mock_completion.side_effect = [
            Timeout("timeout", llm_provider="openai", model="gpt"),
            Timeout("timeout", llm_provider="openai", model="gpt"),
            success,
        ]
        llm = LLM(config=LLMConfig(model_name="gpt-5-nano", num_retries=3))
        prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[])
        result = llm(prompt)
        assert result.message.content == "final answer"
        assert mock_completion.call_count == 3
