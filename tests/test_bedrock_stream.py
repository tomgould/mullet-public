from mullet.core.providers.bedrock_stream import BedrockStreamHandler
import json

def mock_event_stream(events_data):
    for data in events_data:
        # Simulate the actual Bedrock stream format where 'bytes' is a JSON-encoded string
        # The data itself is the payload that would be inside the 'bytes' field
        yield {'chunk': {'bytes': json.dumps(data).encode('utf-8')}}

def test_parse_stream_content_block_start():
    handler = BedrockStreamHandler()
    events = [
        {"content_block_start": {"block_index": 0, "type": "content_block_start", "content_block": {"type": "text", "text": ""}}}
    ]
    chunks = list(handler.parse_stream(mock_event_stream(events)))
    assert len(chunks) == 1
    assert chunks[0].type == "content_block_start"
    assert chunks[0].start_event['block_index'] == 0

def test_parse_stream_content_block_delta():
    handler = BedrockStreamHandler()
    events = [
        {"content_block_delta": {"block_index": 0, "type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}}
    ]
    chunks = list(handler.parse_stream(mock_event_stream(events)))
    assert len(chunks) == 1
    assert chunks[0].type == "content_block_delta"
    assert chunks[0].text == "Hello"

def test_parse_stream_message_stop():
    handler = BedrockStreamHandler()
    events = [
        {"message_stop": {"stop_reason": "end_turn", "stop_sequence": None}}
    ]
    chunks = list(handler.parse_stream(mock_event_stream(events)))
    assert len(chunks) == 1
    assert chunks[0].type == "message_stop"
    assert chunks[0].stop_event['stop_reason'] == "end_turn"

def test_parse_stream_mixed_events():
    handler = BedrockStreamHandler()
    events = [
        {"content_block_start": {"block_index": 0, "type": "content_block_start", "content_block": {"type": "text", "text": ""}}},
        {"content_block_delta": {"block_index": 0, "type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}},
        {"content_block_delta": {"block_index": 0, "type": "content_block_delta", "delta": {"type": "text_delta", "text": " world!"}}},
        {"message_stop": {"stop_reason": "end_turn", "stop_sequence": None}}
    ]
    chunks = list(handler.parse_stream(mock_event_stream(events)))
    assert len(chunks) == 4
    assert chunks[0].type == "content_block_start"
    assert chunks[1].text == "Hello"
    assert chunks[2].text == " world!"
    assert chunks[3].type == "message_stop"

def test_parse_stream_empty_stream():
    handler = BedrockStreamHandler()
    events = []
    chunks = list(handler.parse_stream(mock_event_stream(events)))
    assert len(chunks) == 0

def test_parse_stream_unsupported_event_type():
    handler = BedrockStreamHandler()
    events = [
        {"unsupported_event_type": {"key": "value"}}
    ]
    chunks = list(handler.parse_stream(mock_event_stream(events)))
    assert len(chunks) == 0

def test_parse_stream_non_json_bytes():
    handler = BedrockStreamHandler()
    def bad_stream():
        # This simulates a malformed chunk where bytes are not valid JSON
        yield {'chunk': {'bytes': b'not json'}}
    chunks = list(handler.parse_stream(bad_stream()))
    assert len(chunks) == 0

def test_parse_stream_no_chunk_key():
    handler = BedrockStreamHandler()
    # This simulates an event in the stream that does not have the 'chunk' key at all
    def stream_without_chunk_key():
        yield {"not_a_chunk": {"data": "some data"}}
    chunks = list(handler.parse_stream(stream_without_chunk_key()))
    assert len(chunks) == 0
