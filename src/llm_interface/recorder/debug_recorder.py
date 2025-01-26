import json
import os
import uuid
from datetime import datetime

from llm_interface.model import LlmMessage


class DebugRecorder:
    """Record LLM calls for manual inspection."""

    DEFAULT_OUTPUT_DIR = ".llm_recorder"

    def __init__(self, output_dir=DEFAULT_OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._chat_id_to_filename_cache: dict[str, str] = {}

    def record(self, model: str, messages: list[LlmMessage], *args, **kwargs):
        """Record a LLM completion call."""
        call_id = uuid.uuid4()
        dt = datetime.now()
        call_info = {
            "id": str(call_id),
            "timestamp": dt.isoformat(),
            "model": model,
            "messages": [msg.model_dump() for msg in messages],
        }
        # If stored chat with same chat id exists then overwrite it to prevent duplicates
        chat_id = kwargs.get("chat_id", None)
        if chat_id and chat_id in self._chat_id_to_filename_cache:
            filename = self._chat_id_to_filename_cache[chat_id]
        else:
            filename = f"{dt.strftime('%Y%m%d_%H%M%S')}_{call_id}.json"
        with open(os.path.join(self.output_dir, filename), "w") as f:
            json.dump(call_info, f, indent=2)
        if chat_id:
            self._chat_id_to_filename_cache[kwargs["chat_id"]] = filename
