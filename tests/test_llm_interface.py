import asyncio
from typing import Any

import pytest

from llm_interface import LLMInterface
from llm_interface.model import LlmToolMessage


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
