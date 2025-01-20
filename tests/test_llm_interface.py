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
