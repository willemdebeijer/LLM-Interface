import asyncio
import inspect
import json
import logging
import time
import uuid
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

from llm_interface.exception import LlmException
from llm_interface.llm_handler import AbstractLlmHandler, OpenAiLlmHandler
from llm_interface.model import (
    LlmCompletionMessage,
    LlmCompletionMetadata,
    LlmMessage,
    LlmMultiMessageCompletion,
    LlmSystemMessage,
    LlmToolCall,
    LlmToolMessage,
    LlmToolMessageMetadata,
    LlmUserMessage,
)
from llm_interface.protocol import LlmRepresentable
from llm_interface.recorder import DebugRecorder

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def time_coroutine(coroutine):
    """Await coroutine and return its results and the duration in seconds"""
    start = time.time()
    result = await coroutine
    end = time.time()
    return result, end - start


class LLMInterface:
    """An interface for working with LLMs.

    :param openai_api_key: The OpenAI API key to use. Required if no handler is provided.
    :param handler: A custom LLM handler to use. Will override `openai_api_key`.
    :param verbose: If True, log debug messages.
    :param debug: If True, record all LLM calls and make them available to the viewer.
    :param default_model: The default model to use for LLM calls.
    :param default_temperature: The default temperature to use for LLM calls.
    """

    class NotAllToolsMatchedException(Exception):
        pass

    def __init__(
        self,
        openai_api_key: str | None = None,
        handler: AbstractLlmHandler | None = None,
        default_model: str | None = None,
        default_temperature: float | None = 0.7,
        verbose=True,
        debug=False,
    ):
        if handler:
            self.handler = handler
        elif openai_api_key:
            self.handler: AbstractLlmHandler = OpenAiLlmHandler(openai_api_key)
        else:
            raise Exception("Please provide `openai_api_key` or `handler`")
        self.verbose = verbose
        self.debug = (
            debug  # Will record all LLM calls and make them available to the viewer
        )
        self.recorders = []
        if debug:
            self.recorders.append(DebugRecorder())
        if verbose:
            logger.setLevel(logging.DEBUG)
        self.default_model = default_model
        self.default_temperature = default_temperature

    async def get_completion(
        self,
        messages: Sequence[Union[LlmMessage, dict[str, Any]]],
        model: str | None = None,
        temperature: float | None = None,
        tools: list[Callable] | None = None,
        request_kwargs: dict | None = None,
        **kwargs,
    ) -> LlmCompletionMessage:
        """Call the LLM and return the completion message, including metadata

        :param tools: List of Python functions that the LLM can use.
            Will automatically be converted to a format that can be used by the LLM API.
        :parma request_kwargs: Freeform dict that will be passed to the LLM API
        """
        model = model or self.default_model
        assert model, "Either a default model must be set or a model must be provided"
        temperature = temperature or self.default_temperature
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
            **(request_kwargs or {}),
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

        completion_message: LlmCompletionMessage = await self.handler.call(data)

        for recorder in self.recorders:
            recorder.record(
                model=model, messages=message_objs + [completion_message], **kwargs
            )
        logger.debug(
            f"OpenAI response in {completion_message.metadata.duration_seconds or 0:.2f}s "
            f"has {len(completion_message.content or '')} characters "
            f"and {len(completion_message.tool_calls or [])} tool calls"
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
        model: str | None = None,
        temperature: float | None = None,
        auto_execute_tools: list[Callable] = [],
        non_auto_execute_tools: list[Callable] = [],
        max_depth: int = 16,
        error_on_max_depth: bool = True,
        request_kwargs: dict | None = None,
    ) -> LlmMultiMessageCompletion:
        """Get AI response including handling tool calls. Return final message and list of all new messages (including the final message).

        :param auto_execute_tools: List of Python functions that the LLM can use and execute automatically.
            This can cause multiple LLM calls for a single call of this method.
        :param non_auto_execute_tools: List of Python functions that the LLM can use but should not execute automatically.
            If a non-auto execute tool is called, this method will return the result up to that point.
        :param max_depth: Maximum number of LLM calls to make when calling this method once.
        :param error_on_max_depth: If True, raise an error when the maximum depth is reached.
            Otherwise, return the result up to that point.
        :param request_kwargs: Freeform dict that will be passed to the LLM API, shared with all calls
        """
        start_time = time.time()
        chat_id = str(uuid.uuid4())
        if max_depth < 1:
            raise ValueError("Max depth must be at least 1.")
        new_messages: list[LlmMessage] = []
        for i in range(max_depth):
            completion: LlmCompletionMessage = await self.get_completion(
                messages=[
                    *messages,
                ]
                + [*new_messages],
                model=model,
                temperature=temperature,
                tools=auto_execute_tools + non_auto_execute_tools,
                request_kwargs=request_kwargs,
                chat_id=chat_id,
            )
            new_messages.append(completion)
            if not completion.tool_calls:
                break
            try:
                new_tool_messages = await self.__handle_tool_call_requests(
                    completion.tool_calls, auto_execute_tools
                )
                new_messages.extend(new_tool_messages)
            except self.NotAllToolsMatchedException:
                logger.debug(
                    "Return call with uncompleted tool calls, since not all tools should be auto executed"
                )
                break
        is_hit_max_depth = i >= max_depth - 1
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
            llm_model_version=completion_messages[-1].metadata.llm_model_version,
            llm_model=completion_messages[-1].metadata.llm_model,
        )
        result = LlmMultiMessageCompletion(messages=new_messages, metadata=metadata)
        return result

    async def __handle_tool_call_requests(
        self, tool_calls: list[LlmToolCall], auto_execute_tools: list[Callable]
    ) -> list[LlmMessage]:
        """Execute tool call requests obtained from the LLM."""
        new_messages: list[LlmMessage] = []
        matched_tools = []
        for tool_call in tool_calls:
            tool: Optional[Callable] = next(
                (t for t in auto_execute_tools if t.__name__ == tool_call.name),
                None,
            )
            matched_tools.append(tool)
        all_matched = all(tool is not None for tool in matched_tools)
        if not all_matched:
            raise self.NotAllToolsMatchedException
        async_tasks = []
        for tool, tool_call in zip(matched_tools, tool_calls):
            if not tool:
                raise LlmException("No matched tool")
            if inspect.iscoroutinefunction(tool):
                async_tasks.append((tool_call, tool(**tool_call.arguments)))
                continue
            start = time.time()
            result = tool(**tool_call.arguments)
            wall_time_seconds = time.time() - start
            tool_message = LlmToolMessage(
                content=self._get_llm_repr(result),
                tool_call_id=tool_call.id,
                raw_content=result,
                metadata=LlmToolMessageMetadata(
                    wall_time_seconds=wall_time_seconds, is_async=False
                ),
            )
            new_messages.append(tool_message)
        if async_tasks:
            results = await asyncio.gather(
                *(time_coroutine(task[1]) for task in async_tasks)
            )
            new_messages.extend(
                [
                    LlmToolMessage(
                        content=self._get_llm_repr(result[0]),
                        tool_call_id=task[0].id,
                        raw_content=result[0],
                        metadata=LlmToolMessageMetadata(
                            wall_time_seconds=result[1], is_async=True
                        ),
                    )
                    for result, task in zip(results, async_tasks)
                ]
            )
        return new_messages

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

    @classmethod
    def _get_llm_repr(cls, obj: Any) -> str:
        """Get a string representation of an object that is suitable for LLM use"""
        if isinstance(obj, LlmRepresentable):
            return obj.llm_repr
        return repr(obj)
