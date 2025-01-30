from llm_interface.model.llm import LlmProvider

openai = LlmProvider(name="OpenAI")
groq = LlmProvider(name="Groq")

__all__ = ["openai", "groq"]
