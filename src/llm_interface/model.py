from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, computed_field

from llm_interface.llm import LlmFamily


class LlmSystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str


class LlmUserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str


class LlmToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str
    raw_content: (
        Any  # The raw value that is returned by the tool without converting to a string
    )


class LlmToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class LlmCompletionMetadata(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_seconds: float | None = None
    llm_model_name: str | None = None
    llm_family: LlmFamily | None = None

    @computed_field
    def input_cost_usd(self) -> float | None:
        if (
            (llm_family := self.llm_family)
            and (usd_per_1m_input_tokens := llm_family.usd_per_1m_input_tokens)
            and (input_tokens := self.input_tokens)
        ):
            return usd_per_1m_input_tokens * input_tokens / 1_000_000
        return None

    @computed_field
    def output_cost_usd(self) -> float | None:
        if (
            (llm_family := self.llm_family)
            and (usd_per_1m_output_tokens := llm_family.usd_per_1m_output_tokens)
            and (output_tokens := self.output_tokens)
        ):
            return usd_per_1m_output_tokens * output_tokens / 1_000_000
        return None

    @computed_field
    def cost_usd(self) -> float | None:
        if (input_cost_usd := self.input_cost_usd()) and (
            output_cost_usd := self.output_cost_usd()
        ):
            return input_cost_usd + output_cost_usd
        return None

    model_config = ConfigDict(computed_fields=True)


class LlmCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[list[LlmToolCall]] = None
    metadata: LlmCompletionMetadata


LlmMessage = Union[
    LlmSystemMessage, LlmUserMessage, LlmToolMessage, LlmCompletionMessage
]


class LlmMultiMessageCompletion(BaseModel):
    """A completion that takes multiple messages, e.g. because of tool calls"""

    messages: list[LlmMessage]
    metadata: LlmCompletionMetadata  # The combined metadata, e.g. total tokens, total duration etc

    @property
    def content(self) -> str | None:
        """The content of the final message"""
        if not self.messages:
            return None
        return self.messages[-1].content

    @computed_field
    def llm_call_count(self) -> int:
        return len([m for m in self.messages if isinstance(m, LlmCompletionMessage)])
