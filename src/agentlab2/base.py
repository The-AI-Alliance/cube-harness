"""Base classes for AgentLab2."""

import importlib

from pydantic import BaseModel, model_serializer, model_validator


class TypedBaseModel(BaseModel):
    """
    Base class for Pydantic models that can save and load their type information.

    When serialized, includes `_type` field with the fully qualified class name.
    When deserialized, uses `_type` to instantiate the correct subclass.

    This allows saving/loading configs where the field type is an abstract base class
    but the actual value is a concrete subclass (e.g., AgentConfig -> ReactAgentConfig).
    """

    @model_serializer(mode="wrap")
    def _serialize_with_type(self, handler):
        data = handler(self)
        data["_type"] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return data

    @model_validator(mode="wrap")
    @classmethod
    def _deserialize_with_type(cls, value, handler):
        if isinstance(value, dict) and "_type" in value:
            type_path = value.pop("_type")
            module_name, class_name = type_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            actual_cls = getattr(module, class_name)
            return actual_cls.model_validate(value)
        return handler(value)
