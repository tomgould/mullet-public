"""Handles streaming responses from AWS Bedrock models."""

from typing import Iterator, Dict, Any
from dataclasses import dataclass
import json

@dataclass
class BedrockStreamEvent:
    """Represents a single event in a Bedrock streaming response."""
    type: str
    text: str = ""
    start_event: Dict[str, Any] | None = None
    delta_event: Dict[str, Any] | None = None
    stop_event: Dict[str, Any] | None = None

class BedrockStreamHandler:
    """Parses AWS Bedrock streaming response events."""

    def parse_stream(self, event_stream: Iterator[Dict[str, Any]]) -> Iterator[BedrockStreamEvent]:
        """Parse an AWS Bedrock event stream into BedrockStreamEvent objects."""
        for event in event_stream:
            if 'chunk' in event and 'bytes' in event['chunk']:
                try:
                    payload = json.loads(event['chunk']['bytes'].decode('utf-8'))
                    if 'content_block_start' in payload:
                        yield BedrockStreamEvent(type="content_block_start", start_event=payload['content_block_start'])
                    elif 'content_block_delta' in payload:
                        yield BedrockStreamEvent(type="content_block_delta", text=payload['content_block_delta']['delta']['text'], delta_event=payload['content_block_delta'])
                    elif 'message_stop' in payload:
                        yield BedrockStreamEvent(type="message_stop", stop_event=payload['message_stop'])
                except json.JSONDecodeError:
                    # Handle cases where bytes are not valid JSON
                    continue
