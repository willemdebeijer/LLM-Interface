import inspect
import json
from enum import Enum
from typing import (
    Any,
    Callable,
    List,
    Literal,
    get_args,
    get_origin,
    get_type_hints,
)

from llm_interface.model import (
    LlmCompletionMessage,
    LlmMessage,
    LlmSystemMessage,
    LlmToolMessage,
    LlmUserMessage,
)


def safe_nested_get(data, keys) -> Any | None:
    """Get a nested value from a dictionary/list safely."""
    next_result: Any = data
    for key in keys:
        try:
            next_result: Any = next_result[key]
        except (KeyError, IndexError, TypeError):
            return None
    return next_result


class LlmConversionHelpers:
    """Helper class for converting between different LLM-related formats."""

    @classmethod
    def convert_to_llm_message_obj(cls, message_dict: dict | LlmMessage) -> LlmMessage:
        """Convert a dict to the apropriate LLM message type based on role"""
        if isinstance(message_dict, LlmMessage):
            return message_dict
        role = message_dict.get("role")

        if role == "system":
            return LlmSystemMessage.model_validate(message_dict)
        elif role == "user":
            return LlmUserMessage.model_validate(message_dict)
        elif role == "assistant":
            return LlmCompletionMessage.model_validate(message_dict)
        elif role == "tool":
            values = {**message_dict}
            if "raw_content" not in values:
                values["raw_content"] = None
            return LlmToolMessage.model_validate(values)
        else:
            raise ValueError(f"Unknown message role: {role}")

    @classmethod
    def convert_to_llm_message_dict(cls, message_obj: LlmMessage | dict) -> dict:
        """Convert a LLM message object to a dictionary, leaving out values that should not be sent to the API"""
        if isinstance(message_obj, dict):
            return message_obj
        if isinstance(message_obj, LlmCompletionMessage):
            d: dict[str, Any] = {"role": message_obj.role}
            if message_obj.content is not None:
                d["content"] = message_obj.content
            if message_obj.tool_calls is not None:
                d["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                    for tool_call in message_obj.tool_calls
                ]
            return d
        if isinstance(message_obj, LlmToolMessage):
            return message_obj.model_dump(exclude={"raw_content", "metadata"})
        return message_obj.model_dump()

    @classmethod
    def convert_type_to_json_schema(cls, type_hint: Any) -> dict:
        """Convert Python type hints to JSON Schema types"""
        # Handle basic types
        if type_hint is str:
            return {"type": "string"}
        elif type_hint is int:
            return {"type": "integer"}
        elif type_hint is bool:
            return {"type": "boolean"}

        # Get the origin type (e.g., List from List[str])
        origin = get_origin(type_hint)

        # Handle Lists/Arrays
        if origin is list or origin == List:
            args = get_args(type_hint)
            if not args:
                raise ValueError(
                    "List type must specify its items type (e.g., List[str])"
                )
            item_type = args[0]
            return {
                "type": "array",
                "items": cls.convert_type_to_json_schema(item_type),
            }

        # Handle Dicts/Objects
        # if origin == dict or origin == Dict:
        #     args = get_args(type_hint)
        #     if not args or args[0] != str:
        #         raise ValueError("Dict must use string keys (e.g., Dict[str, Any])")
        #     value_type = args[1]
        #     if hasattr(value_type, '__annotations__'):
        #         # If it's a TypedDict or similar
        #         properties = {}
        #         required = []
        #         for key, t in value_type.__annotations__.items():
        #             properties[key] = convert_type_to_json_schema(t)
        #             required.append(key)  # Assuming all fields required for simplicity
        #         return {
        #             "type": "object",
        #             "properties": properties,
        #             "required": required
        #         }
        #     else:
        #         # If it's a simple Dict[str, something]
        #         return {
        #             "type": "object",
        #             "additionalProperties": convert_type_to_json_schema(value_type)
        #         }

        # Handle Enums
        if isinstance(type_hint, type) and issubclass(type_hint, Enum):
            return {"type": "string", "enum": [e.value for e in type_hint]}

        # Handle Literal types (for enum-like values)
        if hasattr(type_hint, "__origin__") and type_hint.__origin__ is Literal:
            return {"type": "string", "enum": list(type_hint.__args__)}

        # Unsupported type
        raise ValueError(
            f"Unsupported type: {type_hint}. Only str, int, bool, List, Dict, Enum, and Literal are supported."
        )

    @classmethod
    def function_to_tool(cls, func: Callable) -> dict:
        """Convert a Python function to an OpenAI tool definition"""
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""
        type_hints = get_type_hints(func)

        parameters = {"type": "object", "properties": {}, "required": []}

        for param_name, param in sig.parameters.items():
            # Skip *args and **kwargs parameters
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            param_type = type_hints.get(param_name, str)
            param_schema = cls.convert_type_to_json_schema(param_type)

            # Check if parameter has a default value
            if param.default == param.empty:
                parameters["required"].append(param_name)

            # Get parameter description from docstring if available
            # (you might want to use a more sophisticated docstring parser)
            param_doc = ""
            if f":param {param_name}:" in doc:
                str_from_param_doc = doc.split(f":param {param_name}:")[1]
                param_doc = ""
                for line in str_from_param_doc.split("\n"):
                    if line.startswith(":") or line.strip() == "":
                        break
                    param_doc += line.strip() + "\n"
                param_doc = param_doc.strip()

            param_schema["description"] = param_doc
            parameters["properties"][param_name] = param_schema

        lines = doc.split("\n")
        description = ""
        for line in lines:
            if ":param" in line or line.startswith("Args:"):
                break
            description += line + "\n"
        # Remove trailing empty lines
        description = description.rstrip()

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": parameters,
            },
        }
