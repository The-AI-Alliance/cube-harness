import json

from agentlab2.agent import AgentConfig
from agentlab2.benchmark import Benchmark
from agentlab2.core import AL2BaseModel
from agentlab2.environment import EnvironmentConfig
from agentlab2.experiment import Experiment
from tests.conftest import MockAgentConfig, SerializableBenchmark, SerializableEnvConfig


class TestAL2BaseModel:
    """Tests for AL2BaseModel polymorphic serialization/deserialization."""

    def test_serialization_includes_type_field(self):
        """Test that serialization includes _type field with class path."""

        class ConcreteModel(AL2BaseModel):
            value: int = 42

        model = ConcreteModel()
        data = model.model_dump()

        assert "_type" in data
        assert data["_type"].endswith("ConcreteModel")
        assert data["value"] == 42

    def test_deserialization_with_type_field(self):
        """Test that deserialization uses _type to instantiate correct class."""
        data = {
            "_type": "tests.conftest.MockAgentConfig",
            "name": "custom_mock",
        }

        # Deserialize using base class
        result = AgentConfig.model_validate(data)

        assert isinstance(result, MockAgentConfig)
        assert result.name == "custom_mock"

    def test_deserialization_without_type_field(self):
        """Test that deserialization works normally without _type field."""
        data = {"name": "normal_mock"}
        result = MockAgentConfig.model_validate(data)

        assert isinstance(result, MockAgentConfig)
        assert result.name == "normal_mock"

    def test_roundtrip_agent_config(self):
        """Test full round-trip serialization of AgentConfig subclass."""
        original = MockAgentConfig(name="roundtrip_test")
        json_str = original.model_dump_json()
        data = json.loads(json_str)

        # Deserialize using base class
        restored = AgentConfig.model_validate(data)

        assert type(restored) is MockAgentConfig
        assert restored.name == original.name

    def test_roundtrip_environment_config(self):
        """Test full round-trip serialization of EnvironmentConfig subclass."""
        original = SerializableEnvConfig()
        json_str = original.model_dump_json()
        data = json.loads(json_str)

        restored = EnvironmentConfig.model_validate(data)

        assert type(restored) is SerializableEnvConfig

    def test_roundtrip_benchmark(self):
        """Test full round-trip serialization of Benchmark subclass."""
        original = SerializableBenchmark(env_config=SerializableEnvConfig(), metadata={"key": "value"})
        json_str = original.model_dump_json()
        data = json.loads(json_str)

        restored = Benchmark.model_validate(data)

        assert type(restored) is SerializableBenchmark
        assert restored.metadata == {"key": "value"}

    def test_nested_polymorphic_deserialization(self):
        """Test that nested polymorphic models are correctly deserialized."""
        original = SerializableBenchmark(env_config=SerializableEnvConfig())
        json_str = original.model_dump_json()
        data = json.loads(json_str)

        restored = Benchmark.model_validate(data)

        # Both Benchmark and its nested env_config should be correct types
        assert type(restored) is SerializableBenchmark
        assert type(restored.env_config) is SerializableEnvConfig

    def test_experiment_roundtrip(self, tmp_dir):
        """Test full round-trip of Experiment with polymorphic fields."""
        original = Experiment(
            name="test_exp",
            output_dir=tmp_dir,
            agent_config=MockAgentConfig(name="test_agent"),
            benchmark=SerializableBenchmark(env_config=SerializableEnvConfig()),
        )

        # serialize_as_any=True is needed to serialize subclass-specific fields
        json_str = original.model_dump_json(serialize_as_any=True)
        data = json.loads(json_str)
        restored = Experiment.model_validate(data)

        assert restored.name == "test_exp"
        assert type(restored.agent_config) is MockAgentConfig
        assert restored.agent_config.name == "test_agent"
        assert type(restored.benchmark) is SerializableBenchmark
        assert type(restored.benchmark.env_config) is SerializableEnvConfig
