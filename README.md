# LLM Interface

LLMs are awesome but working with them can be tricky. There are a lot of complex frameworks and packages that hide the complexity but obfuscate certain features or make it hard to adapt to the quickly changing advancements in AI.

This is a simple interface to handle the basics.

Features:
- Python method definiton to LLM
- Pretty logging for debugging
- Tool call handling

To add:
- Caching
- Retry
- Metering
- Parallel calls

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

## Using

Run tests by running `pytest` in the root of the project.
