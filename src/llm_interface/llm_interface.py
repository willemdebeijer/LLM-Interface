import asyncio
import inspect
import json
import logging
import time
from collections.abc import Sequence
from enum import Enum
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import aiohttp

from llm_interface.exception import LlmException, RateLimitException
from llm_interface.llm import LlmFamily
from llm_interface.model import (
    LlmCompletionMessage,
    LlmCompletionMetadata,
    LlmMessage,
    LlmMultiMessageCompletion,
    LlmSystemMessage,
    LlmToolCall,
    LlmToolMessage,
    LlmUserMessage,
)
from llm_interface.recorder import DebugRecorder

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class LLMInterface:
    """An interface for working with LLMs."""

    def __init__(self, openai_api_key: str, verbose=True, debug=False):
        self.oai_api_key = openai_api_key
        self.base_url = "https://api.openai.com/v1/"
        self.verbose = verbose
        self.debug = (
            debug  # Will record all LLM calls and make them available to the viewer
        )
        self.recorders = []
        if debug:
            self.recorders.append(DebugRecorder())
        if verbose:
            logger.setLevel(logging.DEBUG)

    async def get_completion(
        self,
        messages: Sequence[Union[LlmMessage, dict[str, Any]]],
        model: str,
        temperature: float = 0.7,
        tools: list[Callable] | None = None,
    ) -> LlmCompletionMessage:
        start_time = time.time()

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

        if self.verbose:
            logger.debug("-" * 64)
            logger.debug(f"Calling OpenAI {model} with {len(messages)} messages")
            logger.debug(
                f"Input messages:\n{self._format_for_log(serialized_messages)}"
            )

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

        end_time = time.time()
        duration = end_time - start_time
        llm_model_name = self.safe_nested_get(result, ("model",))
        llm_family = (
            LlmFamily.get_family_for_model_name(llm_model_name)
            if llm_model_name
            else None
        )
        metadata = LlmCompletionMetadata(
            input_tokens=self.safe_nested_get(result, ("usage", "prompt_tokens")),
            output_tokens=self.safe_nested_get(result, ("usage", "completion_tokens")),
            duration_seconds=duration,
            llm_model_name=llm_model_name,
            llm_family=llm_family,
        )
        completion_message = LlmCompletionMessage(
            content=text, tool_calls=tool_calls, metadata=metadata
        )
        for recorder in self.recorders:
            recorder.record(model=model, messages=message_objs + [completion_message])
        logger.debug(
            f"OpenAI response in {duration:.2f}s has {len(text or '')} characters and {len(tool_calls)} tool calls"
        )
        if self.verbose:
            logger.debug("-" * 64)
        return completion_message

    def _format_for_log(self, obj: Any, prefix: str = "   ") -> str:
        """Format an object for readable logging"""
        try:
            if isinstance(obj, dict) or isinstance(obj, list):
                return "\n".join(
                    prefix + line for line in json.dumps(obj, indent=2).splitlines()
                )
        except Exception:
            pass
        return str(obj)

    async def get_auto_tool_completion(
        self,
        messages: Sequence[Union[LlmMessage, dict[str, Any]]],
        model: str,
        temperature: float = 0.7,
        auto_execute_tools: list[Callable] = [],
        non_auto_execute_tools: list[Callable] = [],
        max_depth: int = 16,
        error_on_max_depth: bool = True,
    ) -> LlmMultiMessageCompletion:
        """Get AI response including handling tool calls. Return final message and list of all new messages (including the final message)."""
        start_time = time.time()
        if max_depth < 1:
            raise ValueError("Max depth must be at least 1.")
        new_messages: list[LlmMessage] = []
        is_hit_max_depth = True
        for i in range(max_depth):
            completion: LlmCompletionMessage = await self.get_completion(
                messages=[
                    *messages,
                ]
                + [*new_messages],
                model=model,
                temperature=temperature,
                tools=auto_execute_tools + non_auto_execute_tools,
            )
            new_messages.append(completion)
            if not completion.tool_calls:
                is_hit_max_depth = False
                break
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
                is_hit_max_depth = False
                break
            async_tasks = []
            for tool, tool_call in zip(matched_tools, completion.tool_calls):
                if not tool:
                    raise LlmException("No matched tool")
                if inspect.iscoroutinefunction(tool):
                    async_tasks.append((tool_call, tool(**tool_call.arguments)))
                    continue
                result = tool(**tool_call.arguments)
                tool_message = LlmToolMessage(
                    content=repr(result), tool_call_id=tool_call.id, raw_content=result
                )
                new_messages.append(tool_message)
            if async_tasks:
                results = await asyncio.gather(*(task[1] for task in async_tasks))
                new_messages.extend(
                    [
                        LlmToolMessage(
                            content=repr(result),
                            tool_call_id=task[0].id,
                            raw_content=result,
                        )
                        for result, task in zip(results, async_tasks)
                    ]
                )
        if is_hit_max_depth:
            logger.error("Max depth reached")
            if error_on_max_depth:
                raise LlmException("Max depth reached")
        completion_messages: list[LlmCompletionMessage] = [
            m for m in new_messages if isinstance(m, LlmCompletionMessage)
        ]
        if len(completion_messages) == 0:
            raise LlmException("No completion messages returned")
        end_time = time.time()
        duration = end_time - start_time

        metadata = LlmCompletionMetadata(
            input_tokens=sum(i.metadata.input_tokens or 0 for i in completion_messages)
            if all(i.metadata.input_tokens is not None for i in completion_messages)
            else None,
            output_tokens=sum(
                i.metadata.output_tokens or 0 for i in completion_messages
            )
            if all(i.metadata.output_tokens is not None for i in completion_messages)
            else None,
            duration_seconds=duration,
            llm_model_name=completion_messages[-1].metadata.llm_model_name,
            llm_family=completion_messages[-1].metadata.llm_family,
        )
        result = LlmMultiMessageCompletion(messages=new_messages, metadata=metadata)
        return result

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
