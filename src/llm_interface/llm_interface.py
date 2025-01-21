import inspect
import json
import logging
from enum import Enum
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Optional,
    get_args,
    get_origin,
    get_type_hints,
)

import aiohttp

from .exception import LlmException, RateLimitException
from .model import (
    LlmCompletionMessage,
    LlmMessage,
    LlmSystemMessage,
    LlmToolCall,
    LlmToolMessage,
    LlmUserMessage,
)

logger = logging.getLogger(__name__)


class LLMInterface:
    """An interface for working with LLMs."""

    def __init__(self, openai_api_key: str, verbose=True):
        self.oai_api_key = openai_api_key
        self.base_url = "https://api.openai.com/v1/"
        self.verbose = verbose

    async def get_completion(
        self,
        messages: list[dict[str, Any] | LlmMessage],
        model: str,
        temperature: float = 0.7,
        tools: list[Callable] | None = None,
    ) -> LlmCompletionMessage:
        if self.verbose:
            logger.debug("-" * 64)
            logger.debug(f"Input messages:\n{messages}")
        logger.debug(f"Calling OpenAI {model} with {len(messages)} messages")
        headers = {"Authorization": f"Bearer {self.oai_api_key}"}
        url = f"{self.base_url}chat/completions"
        # First convert all messages to the Pydantic objects to make sure they're valid, then serialize
        message_objs = [
            self.convert_to_llm_message_obj(message) for message in messages
        ]
        serialized_messages = [
            self.convert_to_llm_message_dict(message) for message in message_objs
        ]

        data = {
            "messages": serialized_messages,
            "model": model,
            "temperature": temperature,
        }
        if tools:
            serialized_tools = [self.function_to_tool(tool) for tool in tools]
            data["tools"] = serialized_tools
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 429:
                    raise RateLimitException("API rate limit exceeded")
                if response.status != 200:
                    error_text = await response.text()
                    raise LlmException(
                        f"OpenAI API status code {response.status}, error: {error_text}"
                    )
                result = await response.json()
        if self.verbose:
            logger.debug(f"OpenAI raw response:\n{result}")
        text = self.safe_nested_get(result, ("choices", 0, "message", "content"))
        raw_tool_calls = self.safe_nested_get(
            result, ("choices", 0, "message", "tool_calls")
        )
        tool_calls = []
        if raw_tool_calls:
            tool_calls = [
                LlmToolCall(
                    id=raw_tool_call["id"],
                    name=raw_tool_call["function"]["name"],
                    arguments=json.loads(raw_tool_call["function"]["arguments"]),
                )
                for raw_tool_call in raw_tool_calls
            ]
        logger.debug(
            f"OpenAI response has {len(text or '')} characters and {len(tool_calls)} tool calls"
        )
        if self.verbose:
            logger.debug("-" * 64)
        return LlmCompletionMessage(content=text, tool_calls=tool_calls)

    def get_auto_tool_completion(
        self,
        messages: list[dict[str, Any] | LlmMessage],
        model: str,
        temperature: float = 0.7,
        auto_execute_tools: list[Callable] = [],
        non_auto_execute_tools: list[Callable] = [],
        max_depth: int = 5,
        error_on_max_depth: bool = True,
    ) -> tuple[LlmCompletionMessage, list[LlmMessage]]:
        """Get AI response including handling tool calls. Return final message and list of all new messages (including the final message)."""
        if max_depth < 1:
            raise ValueError("Max depth must be at least 1.")
        new_messages: list[LlmMessage] = []
        for i in range(max_depth):
            completion: LlmCompletionMessage = self.get_completion(
                messages=messages + new_messages,
                model=model,
                temperature=temperature,
                tools=auto_execute_tools + non_auto_execute_tools,
            )
            new_messages.append(completion)
            if not completion.tool_calls:
                return completion, new_messages
            matched_tools = []
            for tool_call in completion.tool_calls:
                tool: Optional[Callable] = next(
                    (t for t in auto_execute_tools if t.__name__ == tool_call.name),
                    None,
                )
                matched_tools.append(tool)
            all_matched = all(tool is not None for tool in matched_tools)
            if not all_matched:
                logger.debug(
                    "Return call with uncompleted tool calls, since not all tools should be auto executed"
                )
                return completion, new_messages
            for tool, tool_call in zip(matched_tools, completion.tool_calls):
                result = tool(**tool_call.arguments)
                tool_message = LlmToolMessage(
                    content=str(result), tool_call_id=tool_call.id, raw_content=result
                )
                new_messages.append(tool_message)
        logger.error("Max depth reached")
        if error_on_max_depth:
            raise LlmException("Max depth reached")
        return completion, new_messages

    @staticmethod
    def safe_nested_get(data, keys):
        """Get a nested value from a dictionary/list safely."""
        for key in keys:
            try:
                data = data[key]
            except (KeyError, IndexError, TypeError):
                return None
        return data

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
        """Convert a LLM message object to a dictionary"""
        if isinstance(message_obj, dict):
            return message_obj
        if isinstance(message_obj, LlmCompletionMessage):
            d = {"role": message_obj.role}
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
            message_obj.model_dump(exclude={"raw_content"})
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
            param_type = type_hints.get(param_name, str)
            param_schema = cls.convert_type_to_json_schema(param_type)

            # Check if parameter has a default value
            if param.default == param.empty:
                parameters["required"].append(param_name)

            # Get parameter description from docstring if available
            # (you might want to use a more sophisticated docstring parser)
            param_doc = ""
            if f":param {param_name}:" in doc:
                param_doc = doc.split(f":param {param_name}:")[1].split("\n")[0].strip()

            param_schema["description"] = param_doc
            parameters["properties"][param_name] = param_schema

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": doc.split("\n")[0] if doc else "",
                "parameters": parameters,
            },
        }
