"""Search across saved conversation sessions stored as JSONL files."""

import json
import os
from collections import defaultdict

class SessionIndex:
    """In-memory inverted index over a directory of saved chat sessions."""

    def __init__(self, session_dir):
        self.session_dir = session_dir
        self.sessions = []
        self._load_sessions()

    def _load_sessions(self):
        for filename in os.listdir(self.session_dir):
            if filename.endswith(".jsonl"):
                filepath = os.path.join(self.session_dir, filename)
                with open(filepath) as f:
                    session_data = []
                    for line in f:
                        try:
                            entry = json.loads(line)
                            session_data.append(entry)
                        except json.JSONDecodeError:
                            continue # Skip malformed lines
                    self.sessions.append({"filename": filename, "messages": self._extract_messages(session_data)})

    def _extract_messages(self, session_data):
        messages = []
        for entry in session_data:
            if "user_message" in entry:
                messages.append(("user", entry["user_message"]))
            if "assistant_message" in entry:
                messages.append(("assistant", entry["assistant_message"]))
        return messages

    def search(self, query, max_results=5):
        """Search sessions for messages matching the query."""
        query_lower = query.lower()
        results = defaultdict(list)

        for session in self.sessions:
            for role, message in session["messages"]:
                if query_lower in message.lower():
                    results[session["filename"]].append({"role": role, "message": message})

        # Sort results by the number of matches in each session (simple relevance)
        sorted_results = sorted(results.items(), key=lambda item: len(item[1]), reverse=True)

        formatted_results = []
        for filename, matches in sorted_results:
            formatted_results.append({"session_file": filename, "matches": matches})
            if len(formatted_results) >= max_results:
                break

        return formatted_results
