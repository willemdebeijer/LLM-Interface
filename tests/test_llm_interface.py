import pytest

from llm_interface import LLMInterface


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
    messages = [{"role": "user", "content": "Hello"}]
    result = await llm.get_completion(messages=messages, model="gpt-4")

    assert result.content == "Hello, world!"


@pytest.mark.asyncio
async def test_get_auto_tool_completion(monkeypatch):
    llm_interface = LLMInterface(openai_api_key="test_key", verbose=False)

    def get_weather(city: str) -> str:
        """Get the current temperature for a given location

        :param city: City and country e.g. Bogotá, Colombia
        """
        return "25 degrees Celsius"

    responses = [
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
    response_iter = iter(responses)

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
