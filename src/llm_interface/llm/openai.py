from llm_interface.model import LlmModel
from llm_interface.provider import openai

gpt_4o = LlmModel(
    name="gpt-4o",
    provider=openai,
    usd_per_1m_input_tokens=2.5,
    usd_per_1m_output_tokens=10,
)
gpt_4o_mini = LlmModel(
    name="gpt-4o-mini",
    provider=openai,
    usd_per_1m_input_tokens=0.15,
    usd_per_1m_output_tokens=0.6,
)
o1 = LlmModel(
    name="o1", provider=openai, usd_per_1m_input_tokens=15, usd_per_1m_output_tokens=60
)
o1_mini = LlmModel(
    name="o1-mini",
    provider=openai,
    usd_per_1m_input_tokens=3,
    usd_per_1m_output_tokens=12,
)
