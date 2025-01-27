from typing import ClassVar, Optional

from pydantic import BaseModel, ConfigDict


class LlmProvider(BaseModel):
    """An API that gives us access to LLMs.

    Note that the same LLM can be on multiple providers, for example with the open source Llama models.
    """

    name: str

    _all: ClassVar[list["LlmProvider"]] = []

    @classmethod
    def get_all(cls) -> list["LlmProvider"]:
        return cls._all


openai = LlmProvider(name="OpenAI")
groq = LlmProvider(name="Groq")


class LlmFamily(BaseModel):
    """Family of LLMs such as GPT-4o which can have different versions.

    Currently we only track families as versions generally all have the same pricing
    Note that Decimal would be more appropriate than float for money, but we choose to use float for simplicity.
        Costs should only be used as estimates.
    """

    name: str
    provider: LlmProvider
    usd_per_1m_input_tokens: float | None = (
        None  # Currently ignores possible discount for cached tokens
    )
    usd_per_1m_output_tokens: float | None = None

    _families: ClassVar[list["LlmFamily"]] = []

    def __init__(self, **data):
        super().__init__(**data)
        # Add instance to class list when created
        self.__class__._families.append(self)

    @classmethod
    def get_all_families(cls) -> list["LlmFamily"]:
        return cls._families

    @classmethod
    def get_family_for_model_name(
        cls, model_name: str, provider: LlmProvider | None = None
    ) -> Optional["LlmFamily"]:
        for family in cls._families[::-1]:
            if provider and family.provider != provider:
                continue
            if model_name.startswith(family.name):
                return family
        return None

    model_config = ConfigDict(
        exclude=["_families"],
    )


# OpenAI models
gpt_4o = LlmFamily(
    name="gpt-4o",
    provider=openai,
    usd_per_1m_input_tokens=2.5,
    usd_per_1m_output_tokens=10,
)
gpt_4o_mini = LlmFamily(
    name="gpt-4o-mini",
    provider=openai,
    usd_per_1m_input_tokens=0.15,
    usd_per_1m_output_tokens=0.6,
)
o1 = LlmFamily(
    name="o1", provider=openai, usd_per_1m_input_tokens=15, usd_per_1m_output_tokens=60
)
o1_mini = LlmFamily(
    name="o1-mini",
    provider=openai,
    usd_per_1m_input_tokens=3,
    usd_per_1m_output_tokens=12,
)

# Groq models
# groq_llama_3_3_70b_specdec_8k = LlmFamily(
#     name="llama-3.3-"
# )
