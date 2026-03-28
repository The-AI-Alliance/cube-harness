import json

from cube_harness.agent import AgentConfig
from cube_harness.core import TypedBaseModel
from cube_harness.experiment import Experiment
from cube_harness.legacy import Benchmark
from tests.conftest import MockAgentConfig, MockToolConfig, SerializableBenchmark


class TestAL2BaseModel:
    """Tests for AL2BaseModel polymorphic serialization/deserialization."""

    def test_serialization_includes_type_field(self):
        """Test that serialization includes _type field with class path."""

        class ConcreteModel(TypedBaseModel):
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

    def test_nested_polymorphic_deserialization(self):
        """Test that nested polymorphic models are correctly deserialized."""
        original = SerializableBenchmark(tool_config=MockToolConfig())
        json_str = original.model_dump_json()
        data = json.loads(json_str)

        restored = Benchmark.model_validate(data)

        # Both Benchmark and its nested env_config should be correct types
        assert type(restored) is SerializableBenchmark
        assert type(restored.tool_config) is MockToolConfig

    def test_experiment_roundtrip(self, tmp_dir):
        """Test full round-trip of Experiment with polymorphic fields."""
        original = Experiment(
            name="test_exp",
            output_dir=tmp_dir,
            agent_config=MockAgentConfig(name="test_agent"),
            benchmark=SerializableBenchmark(tool_config=MockToolConfig()),
        )

        # serialize_as_any=True is needed to serialize subclass-specific fields
        json_str = original.model_dump_json(serialize_as_any=True)
        data = json.loads(json_str)
        restored = Experiment.model_validate(data)

        assert restored.name == "test_exp"
        assert type(restored.agent_config) is MockAgentConfig
        assert restored.agent_config.name == "test_agent"
        assert type(restored.benchmark) is SerializableBenchmark
        assert type(restored.benchmark.tool_config) is MockToolConfig
