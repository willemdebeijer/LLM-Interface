import asyncio
import inspect
import json
import logging
import time
import uuid
from collections.abc import Sequence
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)

from llm_interface.exception import LlmException
from llm_interface.helpers import LlmConversionHelpers
from llm_interface.llm_handler import AbstractLlmHandler, OpenAiLlmHandler
from llm_interface.model import (
    LlmCompletionMessage,
    LlmCompletionMetadata,
    LlmMessage,
    LlmMultiMessageCompletion,
    LlmToolCall,
    LlmToolMessage,
    LlmToolMessageMetadata,
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
    :param default_label: A default label for chats. This will be shown in the viewer sidebar.
    :param default_model: The default model to use for LLM calls.
    :param default_temperature: The default temperature to use for LLM calls.
    """

    class NotAllToolsMatchedException(Exception):
        pass

    def __init__(
        self,
        openai_api_key: str | None = None,
        handler: AbstractLlmHandler | None = None,
        default_label: str | None = None,
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
        self.default_label = default_label
        self.default_model = default_model
        self.default_temperature = default_temperature
        self._completion_metadata: list[LlmCompletionMetadata] = []

    @property
    def total_calls(self) -> int:
        """Total number of LLM API calls made"""
        return len(self._completion_metadata)

    @property
    def total_input_tokens(self) -> int:
        """Total number of input tokens across all calls"""
        return sum(m.input_tokens or 0 for m in self._completion_metadata)

    @property
    def total_output_tokens(self) -> int:
        """Total number of output tokens across all calls"""
        return sum(m.output_tokens or 0 for m in self._completion_metadata)

    @property
    def total_input_cost_usd(self) -> float:
        """Total cost in USD for input tokens across all calls"""
        return sum(m._input_cost_usd or 0.0 for m in self._completion_metadata)

    @property
    def total_output_cost_usd(self) -> float:
        """Total cost in USD for output tokens across all calls"""
        return sum(m._output_cost_usd or 0.0 for m in self._completion_metadata)

    @property
    def total_cost_usd(self) -> float:
        """Total cost in USD across all calls (input + output)"""
        return self.total_input_cost_usd + self.total_output_cost_usd

    @property
    def has_untracked_costs(self) -> bool:
        """Indicates if there are any completions where costs could not be tracked.

        This can happen if the model information is missing or if the cost calculation failed.
        When this is True, the total cost values might be incomplete.
        """
        return any(
            m.input_cost_usd is None or m.output_cost_usd is None
            for m in self._completion_metadata
        )

    async def get_completion(
        self,
        messages: Sequence[Union[LlmMessage, dict[str, Any]]],
        model: str | None = None,
        temperature: float | None = None,
        tools: list[Callable] | None = None,
        request_kwargs: dict | None = None,
        label: str | None = None,
        **kwargs,
    ) -> LlmCompletionMessage:
        """Call the LLM and return the completion message, including metadata

        :param tools: List of Python functions that the LLM can use.
            Will automatically be converted to a format that can be used by the LLM API.
        :param request_kwargs: Freeform dict that will be passed to the LLM API
        :param label: A label the chat. This will be shown in the sidebar in the viewer.
        """
        model = model or self.default_model
        assert model, "Either a default model must be set or a model must be provided"
        temperature = temperature or self.default_temperature
        label = label or self.default_label
        # First convert all messages to the Pydantic objects to make sure they're valid, then serialize
        message_objs = [
            LlmConversionHelpers.convert_to_llm_message_obj(message)
            for message in messages
        ]
        serialized_messages = [
            LlmConversionHelpers.convert_to_llm_message_dict(message)
            for message in message_objs
        ]

        data = {
            "messages": serialized_messages,
            "model": model,
            "temperature": temperature,
            **(request_kwargs or {}),
        }
        if tools:
            serialized_tools = [
                LlmConversionHelpers.function_to_tool(tool) for tool in tools
            ]
            data["tools"] = serialized_tools

        if self.verbose:
            logger.debug("-" * 64)
            logger.debug(f"Calling OpenAI {model} with {len(messages)} messages")
            logger.debug(
                f"Input messages:\n{self._format_for_log(serialized_messages)}"
            )

        completion_message: LlmCompletionMessage = await self.handler.call(
            data, **kwargs
        )

        # Cache the completion metadata
        if completion_message.metadata:
            self._completion_metadata.append(completion_message.metadata)

        for recorder in self.recorders:
            recorder.record(
                model=model,
                messages=message_objs + [completion_message],
                label=label,
                serialized_tools=serialized_tools,
                **kwargs,
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
        label: str | None = None,
        **kwargs,
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
        :param label: A label the chat. This will be shown in the sidebar in the viewer.
        """
        start_time = time.time()
        chat_id = str(uuid.uuid4())
        if max_depth < 1:
            raise ValueError("Max depth must be at least 1.")
        new_messages: list[LlmMessage] = []
        for i in range(max_depth):
            if i != 0:
                kwargs["_preprocessing_duration_seconds"] = time.time() - start_time
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
                label=label,
                **kwargs,
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
    def _get_llm_repr(cls, obj: Any) -> str:
        """Get a string representation of an object that is suitable for LLM use"""
        if isinstance(obj, LlmRepresentable):
            return obj.llm_repr
        return repr(obj)
