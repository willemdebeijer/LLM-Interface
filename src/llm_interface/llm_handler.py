import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from llm_interface.exception import LlmException, RateLimitException
from llm_interface.helpers import safe_nested_get
from llm_interface.model import (
    LlmCompletionMessage,
    LlmCompletionMetadata,
    LlmModel,
    LlmProvider,
    LlmToolCall,
)
from llm_interface.provider import openai


class AbstractLlmHandler(ABC):
    """A base class for handling actual LLM API calls"""

    @abstractmethod
    async def call(self, data: dict[str, Any]) -> LlmCompletionMessage:
        """Handle actual API call.

        :param data: A dict containing the LLM call data such as `messages`, `model` and `temperature`
        """


class OpenAiLlmHandler(AbstractLlmHandler):
    """A handler for OpenAI LLM API calls.

    Can be used with different providers that use the OpenaI API spec by providing a different base URL.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1/",
        provider: LlmProvider = openai,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.provider = provider

    async def __make_api_call(self, url: str, headers: dict, data: dict) -> dict:
        """Make the actual API call to OpenAI.

        :param url: The API endpoint URL
        :param headers: Request headers
        :param data: Request data
        :return: JSON response from the API
        :raises: RateLimitException, LlmException
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 429:
                    raise RateLimitException("API rate limit exceeded")
                if response.status != 200:
                    error_text = await response.text()
                    raise LlmException(
                        f"OpenAI API status code {response.status}, error: {error_text}"
                    )
                return await response.json()

    async def call(self, data: dict[str, Any], **kwargs) -> LlmCompletionMessage:
        """Make an LLM API call to OpenAI.

        :param data: A dict containing the LLM call data such as `messages`, `model` and `temperature`
        :return: A LlmCompletionMessage object containing the LLM response
        """
        start_time = time.time()

        # O1 system prompt requires role 'developer', search for "o" + digit in model name
        is_o_model = "model" in data and bool(re.search(r"o\d", data["model"]))
        if is_o_model and data["messages"][0]["role"] != "developer":
            data["messages"][0]["role"] = "developer"
        # Temperature is not supported for O models
        if is_o_model and "temperature" in data:
            del data["temperature"]

        headers = {"Authorization": f"Bearer {self.api_key}"}
        slash = "/" if not self.base_url.endswith("/") else ""
        url = f"{self.base_url}{slash}chat/completions"

        result = await self.__make_api_call(url, headers, data)

        text = safe_nested_get(result, ("choices", 0, "message", "content"))
        raw_tool_calls = safe_nested_get(
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
        end_to_end_duration = None
        if preprocessing_duration_seconds := kwargs.get(
            "_preprocessing_duration_seconds"
        ):
            end_to_end_duration = duration + preprocessing_duration_seconds
        llm_model_version = safe_nested_get(result, ("model",))
        llm_model = (
            LlmModel.get_model_for_model_name(llm_model_version, provider=self.provider)
            if llm_model_version
            else None
        )
        metadata = LlmCompletionMetadata(
            input_tokens=safe_nested_get(result, ("usage", "prompt_tokens")),
            output_tokens=safe_nested_get(result, ("usage", "completion_tokens")),
            duration_seconds=duration,
            end_to_end_duration_seconds=end_to_end_duration,
            llm_model_version=llm_model_version,
            llm_model=llm_model,
        )
        completion_message = LlmCompletionMessage(
            content=text, tool_calls=tool_calls, metadata=metadata
        )

        return completion_message
