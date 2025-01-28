from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, computed_field

from llm_interface.llm import LlmModel


class LlmSystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str


class LlmUserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str


class LlmToolMessageMetadata(BaseModel):
    wall_time_seconds: float | None = (
        None  # Total time from starting tool to completing it. Might be impacted by multiple simultaneous async calls
    )
    is_async: bool | None = None


class LlmToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str
    raw_content: (
        Any  # The raw value that is returned by the tool without converting to a string
    )
    metadata: LlmToolMessageMetadata = Field(default_factory=LlmToolMessageMetadata)

    # Allow arbitrary types for raw_content
    model_config = {"arbitrary_types_allowed": True}


class LlmToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class LlmCompletionMetadata(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_seconds: float | None = None
    llm_model_version: str | None = (
        None  # Exact checkpoint/version, generally includes release date for OpenAI and is returned by the API
    )
    llm_model: LlmModel | None = None

    @property
    def _input_cost_usd(self) -> float | None:
        if (
            (llm_model := self.llm_model)
            and (usd_per_1m_input_tokens := llm_model.usd_per_1m_input_tokens)
            and (input_tokens := self.input_tokens)
        ):
            return usd_per_1m_input_tokens * input_tokens / 1_000_000
        return None

    @computed_field
    def input_cost_usd(self) -> float | None:
        return self._input_cost_usd

    @property
    def _output_cost_usd(self) -> float | None:
        if (
            (llm_model := self.llm_model)
            and (usd_per_1m_output_tokens := llm_model.usd_per_1m_output_tokens)
            and (output_tokens := self.output_tokens)
        ):
            return usd_per_1m_output_tokens * output_tokens / 1_000_000
        return None

    @computed_field
    def output_cost_usd(self) -> float | None:
        return self._output_cost_usd

    @computed_field
    def cost_usd(self) -> float | None:
        if (input_cost_usd := self._input_cost_usd) and (
            output_cost_usd := self._output_cost_usd
        ):
            return input_cost_usd + output_cost_usd
        return None


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
    def completion_messages(self) -> list[LlmCompletionMessage]:
        return [m for m in self.messages if isinstance(m, LlmCompletionMessage)]

    @property
    def completion_message(self) -> LlmCompletionMessage | None:
        """The final message of the completion"""
        if len(self.completion_messages) == 0:
            return None
        return self.completion_messages[-1]

    @property
    def content(self) -> str | None:
        """The content of the final message"""
        if completion_message := self.completion_message:
            return completion_message.content
        return None

    @property
    def tool_calls(self) -> list[LlmToolCall] | None:
        if completion_message := self.completion_message:
            return completion_message.tool_calls
        return None

    @computed_field
    def llm_call_count(self) -> int:
        return len([m for m in self.messages if isinstance(m, LlmCompletionMessage)])
