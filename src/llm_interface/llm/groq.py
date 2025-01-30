from llm_interface.model import LlmModel
from llm_interface.provider import groq

groq_llama_3_3_70b_specdec_8k = LlmModel(
    name="llama-3.3-70b-specdec",
    provider=groq,
    usd_per_1m_input_tokens=0.59,
    usd_per_1m_output_tokens=0.99,
)

groq_llama_3_3_70b_versatile_128k = LlmModel(
    name="llama-3.3-70b-versatile",
    provider=groq,
    usd_per_1m_input_tokens=0.59,
    usd_per_1m_output_tokens=0.79,
)

groq_deepseek_r1_distill_llama_70b = LlmModel(
    name="deepseek-r1-distill-llama-70b",
    provider=groq,
    usd_per_1m_input_tokens=0.75,
    usd_per_1m_output_tokens=0.99,
)
