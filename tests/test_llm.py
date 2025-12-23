"""Tests for agentlab2.llm module."""

import json
from unittest.mock import MagicMock, patch

from litellm import Message

from agentlab2.llm import LLM, LLMCall, LLMConfig, Prompt


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

    @patch("agentlab2.llm.completion_with_retries")
    def test_llm_call(self, mock_completion, sample_llm_config, sample_prompt):
        """Test LLM call with mocked completion."""
        mock_message = MagicMock()
        mock_message.content = "Hello! How can I help?"
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_completion.return_value = mock_response

        llm = LLM(config=sample_llm_config)
        result = llm(sample_prompt)

        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5-nano"
        assert call_kwargs["temperature"] == 0.7
        assert result.content == "Hello! How can I help?"

    @patch("agentlab2.llm.completion_with_retries")
    def test_llm_call_with_tools(self, mock_completion, sample_llm_config):
        """Test LLM call with tools."""
        mock_message = MagicMock()
        mock_message.tool_calls = [MagicMock(function=MagicMock(name="search", arguments='{"query": "test"}'))]
        mock_message.content = "Tool call made."
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_completion.return_value = mock_response

        tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        prompt = Prompt(messages=[{"role": "user", "content": "Search"}], tools=tools)

        llm = LLM(config=sample_llm_config)
        result = llm(prompt)

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["tools"] == tools
        assert result.tool_calls is not None
        assert result.content == "Tool call made."


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
