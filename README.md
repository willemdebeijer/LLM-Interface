# LLM Interface

LLMs are awesome but working with them can be tricky. There are a lot of complex frameworks and packages that hide the complexity but obfuscate certain features or make it hard to adapt to the quickly changing advancements in AI.

This is a simple interface to handle the basics.

Features:
- Pretty logging for debugging
- Tool call handling
- Python method definiton to LLM
- Metering (e.g. duration, costs, tokens etc)

To add:
- Caching
- Retry
- Simple parallel/batch calls
- Other LLM providers

## Examples

Simplest example:
```python
from llm_interface import LLMInterface

llm_interface = LLMInterface(openai_api_key="YOUR_OPENAI_API_KEY")

simple_result = await llm_interface.get_completion(
    messages=[
        {"role": "user", "content": "What is the capital of the Netherlands?"},
    ],
    model="gpt-4o-mini",
)
simple_result.content
#> The capital of the Netherlands is Amsterdam.
```

### Tool calls

LLMInterface allows you to provide functions as tools to the LLM. It also takes care of the conversion of the function signature to JSON and the execution of the tool.
```python
def get_weather(city: str) -> str:
    """Get the current temperature for a given location
    
    :param city: City and country e.g. Bogotá, Colombia
    """
    return "25 degrees Celsius"

tool_call_result = await llm_interface.get_auto_tool_completion(
    messages=[
        {"role": "user", "content": "What is the weather in Amsterdam?"},
    ],
    model="gpt-4o-mini",
    auto_execute_tools = [get_weather],
)
tool_call_result.content
#> The current temperature in Amsterdam is 25 degrees Celsius.
```

## Using

Run tests by running `pytest` in the root of the project.
