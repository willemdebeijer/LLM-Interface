import json
from unittest.mock import mock_open, patch

import pytest

from llm_interface.model import LlmCompletionMessage, LlmUserMessage
from llm_interface.recorder.debug_recorder import DebugRecorder


@pytest.fixture
def recorder():
    return DebugRecorder(output_dir="test_dir")


@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open)
@patch("uuid.uuid4")
def test_debug_recorder_record(mock_uuid, mock_file, mock_makedirs, recorder):
    # Setup test data
    mock_uuid.return_value = "test-uuid"
    messages = [
        LlmUserMessage(role="user", content="What is 2+2?"),
        LlmCompletionMessage(role="assistant", content="The sum of 2+2 is 4."),
    ]

    # Record the conversation
    recorder.record(
        model="gpt-4", messages=messages, label="math_question", chat_id="test-chat-123"
    )

    # Get and verify the recorded data
    mock_file_handle = mock_file()
    written_str = "".join(call[0][0] for call in mock_file_handle.write.call_args_list)
    written_data = json.loads(written_str)

    # Verify core message data
    assert written_data["model"] == "gpt-4"
    assert written_data["label"] == "math_question"
    assert written_data["id"] == "test-uuid"

    # Verify messages
    assert len(written_data["messages"]) == 2
    assert written_data["messages"][0]["role"] == "user"
    assert written_data["messages"][0]["content"] == "What is 2+2?"
    assert written_data["messages"][1]["role"] == "assistant"
    assert written_data["messages"][1]["content"] == "The sum of 2+2 is 4."
