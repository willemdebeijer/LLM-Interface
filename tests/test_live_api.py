import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Coroutine, TypeVar

import pytest
from dotenv import load_dotenv

from llm_interface import LLMInterface

load_dotenv()


T = TypeVar("T")


def run_coroutine_sync(coroutine: Coroutine[Any, Any, T], timeout: float = 30) -> T:
    """Run any coroutine (async function) synchronously (blocking)
    Works in any situation, including Jupyter Notebooks
    """

    def run_in_new_loop():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coroutine)
        finally:
            new_loop.close()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    if threading.current_thread() is threading.main_thread():
        if not loop.is_running():
            return loop.run_until_complete(coroutine)
        else:
            with ThreadPoolExecutor() as pool:
                future = pool.submit(run_in_new_loop)
                return future.result(timeout=timeout)
    else:
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result()


class Task(Enum):
    SIMPLE = "simple"
    SYSTEM_PROMPT = "system_prompt"
    TOOL_CALL = "tool_call"


def get_weather(city: str) -> str:
    """Get the weather in a city.

    :param city: The name of the city
    """
    return "25 degrees Celsius"


@pytest.mark.api
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,test",
    [
        ("gpt-4o", Task.SIMPLE),
        ("gpt-4o", Task.SYSTEM_PROMPT),
        ("gpt-4o", Task.TOOL_CALL),
        ("o3-mini", Task.SIMPLE),
        ("o3-mini", Task.SYSTEM_PROMPT),
        ("o3-mini", Task.TOOL_CALL),
    ],
)
async def test_oai_model(model: str, test: Task):
    """Test OpenAI model capabilities."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    oai_llm_interface = LLMInterface(
        os.getenv("OPENAI_API_KEY"), default_temperature=None, debug=True, verbose=False
    )

    if test == Task.SIMPLE:
        res = await oai_llm_interface.get_completion(
            messages=[
                {"role": "user", "content": "What's the capital of the Netherlands?"}
            ],
            model=model,
        )
        assert res.content is not None
        assert "Amsterdam" in res.content
    elif test == Task.SYSTEM_PROMPT:
        res = await oai_llm_interface.get_completion(
            messages=[
                {"role": "system", "content": "Be helpful and concise"},
                {"role": "user", "content": "What's the capital of the Netherlands?"},
            ],
            model=model,
        )
        assert res.content is not None
        assert "Amsterdam" in res.content
    elif test == Task.TOOL_CALL:
        res = await oai_llm_interface.get_auto_tool_completion(
            messages=[{"role": "user", "content": "What's the weather in Amsterdam?"}],
            auto_execute_tools=[get_weather],
            model=model,
        )
        assert res.content is not None
        assert res.tool_calls is not None
        assert "25" in res.content
    else:
        raise ValueError(f"Test not supported: {test}")
