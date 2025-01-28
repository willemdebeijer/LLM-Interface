import asyncio
from typing import Any

import pytest

from llm_interface import LLMInterface
from llm_interface.model import LlmCompletionMessage, LlmToolMessage
from llm_interface.model.llm import LlmModel, openai


class MockResponse:
    def __init__(self, data, status):
        self._data = data
        self.status = status

    async def json(self):
        return self._data

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self


@pytest.mark.asyncio
async def test_get_completion(monkeypatch):
    """Test the simplest possible completion."""
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "Hello, world!",
                }
            }
        ]
    }

    mock = MockResponse(mock_response, 200)
    monkeypatch.setattr("aiohttp.ClientSession.post", lambda *args, **kwargs: mock)

    llm = LLMInterface(openai_api_key="test_key", verbose=False)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Hello"}]
    result = await llm.get_completion(messages=messages, model="gpt-4")

    assert result.content == "Hello, world!"
    assert result.metadata.input_tokens is None
    assert result.metadata.output_tokens is None


@pytest.mark.asyncio
async def test_get_completion_with_metadata(monkeypatch):
    """Test the completion uses the metadata."""

    # Create fake model provider
    llm_model = LlmModel(
        name="gpt-4o",
        provider=openai,
        usd_per_1m_input_tokens=0.1,
        usd_per_1m_output_tokens=1.0,
    )

    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "Hello, world!",
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "model": "gpt-4o-v1",
    }

    mock = MockResponse(mock_response, 200)
    monkeypatch.setattr("aiohttp.ClientSession.post", lambda *args, **kwargs: mock)

    llm = LLMInterface(openai_api_key="test_key", verbose=False)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Hello"}]
    result = await llm.get_completion(messages=messages, model="gpt-4o")

    assert result.content == "Hello, world!"
    assert result.metadata.input_tokens == 10
    assert result.metadata.output_tokens == 5
    assert result.metadata.duration_seconds or 0 > 0
    assert result.metadata.llm_model_version == "gpt-4o-v1"
    assert result.metadata.llm_model == llm_model
    assert result.metadata.input_cost_usd or 0 > 0
    assert result.metadata.output_cost_usd or 0 > 0
    assert result.metadata.cost_usd or 0 > 0


mock_weather_responses = [
    {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Amsterdam"}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    },
    {
        "choices": [
            {
                "message": {
                    "content": "The current weather in Amsterdam is 25 degrees Celsius.",
                    "tool_calls": None,
                }
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 7},
    },
]


@pytest.mark.asyncio
async def test_get_auto_tool_completion(monkeypatch):
    """Test LLM completion with a tool call."""
    llm_interface = LLMInterface(openai_api_key="test_key", verbose=False)

    def get_weather(city: str) -> str:
        """Get the current temperature for a given location

        :param city: City and country e.g. Bogotá, Colombia
        """
        return "25 degrees Celsius"

    response_iter = iter(mock_weather_responses)

    def mock_post(*args, **kwargs):
        return MockResponse(next(response_iter), 200)

    monkeypatch.setattr("aiohttp.ClientSession.post", mock_post)

    tool_call_result = await llm_interface.get_auto_tool_completion(
        messages=[
            {"role": "user", "content": "What is the weather in Amsterdam?"},
        ],
        model="gpt-4o-mini",
        auto_execute_tools=[get_weather],
    )

    assert (
        tool_call_result.content
        == "The current weather in Amsterdam is 25 degrees Celsius."
    )
    assert tool_call_result.llm_call_count == 2
    assert tool_call_result.metadata.input_tokens == 10 + 15
    assert tool_call_result.metadata.output_tokens == 5 + 7

    first_completion_message = tool_call_result.messages[0]
    tool_message = tool_call_result.messages[1]
    second_completion_message = tool_call_result.messages[2]
    assert isinstance(first_completion_message, LlmCompletionMessage)
    assert first_completion_message.tool_calls is not None
    assert len(first_completion_message.tool_calls) > 0
    assert isinstance(tool_message, LlmToolMessage)
    assert tool_message.tool_call_id == "call_1"
    assert (
        isinstance(tool_message.metadata.is_async, bool)
        and not tool_message.metadata.is_async
    )
    assert isinstance(second_completion_message, LlmCompletionMessage)
    assert second_completion_message.metadata.input_tokens == 15
    assert second_completion_message.metadata.output_tokens == 7


@pytest.mark.asyncio
async def test_get_async_auto_tool_completion(monkeypatch):
    """Test LLM completion with an async tool call."""
    llm_interface = LLMInterface(openai_api_key="test_key", verbose=False)

    async def get_weather(city: str) -> str:
        """Get the current temperature for a given location

        :param city: City and country e.g. Bogotá, Colombia
        """
        await asyncio.sleep(0.001)
        return "25 degrees Celsius"

    response_iter = iter(mock_weather_responses)

    def mock_post(*args, **kwargs):
        return MockResponse(next(response_iter), 200)

    monkeypatch.setattr("aiohttp.ClientSession.post", mock_post)

    tool_call_result = await llm_interface.get_auto_tool_completion(
        messages=[
            {"role": "user", "content": "What is the weather in Amsterdam?"},
        ],
        model="gpt-4o-mini",
        auto_execute_tools=[get_weather],
    )

    assert (
        tool_call_result.content
        == "The current weather in Amsterdam is 25 degrees Celsius."
    )
    tool_message = next(
        (m for m in tool_call_result.messages if isinstance(m, LlmToolMessage)), None
    )
    assert tool_message is not None
    assert tool_message.content == repr("25 degrees Celsius")
    assert tool_call_result.llm_call_count == 2
    tool_message = tool_call_result.messages[1]
    assert isinstance(tool_message, LlmToolMessage)
    assert tool_message.metadata.is_async


@pytest.mark.asyncio
async def test_get_auto_tool_completion_non_auto_tool(monkeypatch):
    """Test tool call that should not be automatically excuted."""
    llm_interface = LLMInterface(openai_api_key="test_key", verbose=False)

    def get_weather(city: str) -> str:
        raise Exception("Should not be auto executed")

    response_iter = iter(mock_weather_responses)

    def mock_post(*args, **kwargs):
        return MockResponse(next(response_iter), 200)

    monkeypatch.setattr("aiohttp.ClientSession.post", mock_post)

    tool_call_result = await llm_interface.get_auto_tool_completion(
        messages=[
            {"role": "user", "content": "What is the weather in Amsterdam?"},
        ],
        model="gpt-4o-mini",
        non_auto_execute_tools=[get_weather],
    )

    # There should be only a completion with the tool call
    assert len(tool_call_result.messages) == 1


@pytest.mark.asyncio
async def test_get_auto_tool_completion_max_depth(monkeypatch):
    """Test that we stop at the correct amount of calls if the LLM keeps calling a tool."""
    llm_interface = LLMInterface(openai_api_key="test_key", verbose=False)

    def get_weather(city: str) -> str:
        """Get the current temperature for a given location

        :param city: City and country e.g. Bogotá, Colombia
        """
        return "25 degrees Celsius"

    def mock_post(*args, **kwargs):
        # Only ever return first response with tool call
        return MockResponse(mock_weather_responses[0], 200)

    monkeypatch.setattr("aiohttp.ClientSession.post", mock_post)

    tool_call_result = await llm_interface.get_auto_tool_completion(
        messages=[
            {"role": "user", "content": "What is the weather in Amsterdam?"},
        ],
        model="gpt-4o-mini",
        auto_execute_tools=[get_weather],
        max_depth=5,
        error_on_max_depth=False,
    )

    assert (
        len(
            [
                m
                for m in tool_call_result.messages
                if isinstance(m, LlmCompletionMessage)
            ]
        )
        == 5
    )
