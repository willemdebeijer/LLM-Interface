from typing import Any, Literal, Optional, Union

from pydantic import BaseModel


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


class LlmCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[list[LlmToolCall]] = None


LlmMessage = Union[
    LlmSystemMessage, LlmUserMessage, LlmToolMessage, LlmCompletionMessage
]
