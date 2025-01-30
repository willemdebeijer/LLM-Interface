from typing import ClassVar, Optional

from pydantic import BaseModel


class LlmProvider(BaseModel):
    """An API that gives us access to LLMs.

    Note that the same LLM can be on multiple providers, for example with the open source Llama models.
    """

    name: str

    _all: ClassVar[list["LlmProvider"]] = []

    @classmethod
    def get_all(cls) -> list["LlmProvider"]:
        return cls._all

    @property
    def llm_models(self) -> list["LlmModel"]:
        """Get all registered LLMs for this provider."""
        return LlmModel.get_all_for_provider(self)


class LlmModel(BaseModel):
    """A LLMs model such as GPT-4o which can have different checkpoints/versions.

    Currently we only track major models as versions generally all have the same pricing
    Note that Decimal would be more appropriate than float for money, but we choose to use float for simplicity.
        Costs should only be used as estimates.
    """

    name: str
    provider: LlmProvider
    usd_per_1m_input_tokens: float | None = (
        None  # Currently ignores possible discount for cached tokens
    )
    usd_per_1m_output_tokens: float | None = None

    _all: ClassVar[list["LlmModel"]] = []

    def __init__(self, **data):
        super().__init__(**data)
        # Add instance to class list when created
        self.__class__._all.append(self)

    @classmethod
    def get_all(cls) -> list["LlmModel"]:
        return cls._all

    @classmethod
    def get_model_for_model_name(
        cls, model_name: str, provider: LlmProvider | None = None
    ) -> Optional["LlmModel"]:
        for family in cls._all[::-1]:
            if provider and family.provider != provider:
                continue
            if model_name.startswith(family.name):
                return family
        return None

    @classmethod
    def get_all_for_provider(cls, provider: LlmProvider) -> list["LlmModel"]:
        return [model for model in cls._all if model.provider == provider]
