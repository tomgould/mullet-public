import json
import shutil
import unittest
from pathlib import Path

from mullet.core.tools.session_search import SessionIndex


class TestSessionSearch(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("test_session_dir")
        self.test_dir.mkdir(exist_ok=True)
        self.session_dir = self.test_dir / "sessions"
        self.session_dir.mkdir(exist_ok=True)

        # Create dummy session files
        self._create_session_file("session_1.jsonl", [
            {"user_message": "Hello, how are you?"},
            {"assistant_message": "I'm fine, thank you."},
            {"user_message": "What can you do?"}
        ])
        self._create_session_file("session_2.jsonl", [
            {"user_message": "Tell me about Python."},
            {"assistant_message": "Python is a programming language."}
        ])
        self._create_session_file("session_3.jsonl", [
            {"assistant_message": "No user message here."}
        ])
        self._create_session_file("empty_session.jsonl", [])
        self._create_session_file("malformed_session.jsonl", [
            {"user_message": "Valid line"},
            "this is not json",
            {"assistant_message": "Another valid line"}
        ])


    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_session_file(self, filename, data):
        filepath = self.session_dir / filename
        with open(filepath, "w") as f:
            for entry in data:
                if isinstance(entry, dict):
                    f.write(json.dumps(entry) + "\n")
                else:
                    f.write(str(entry) + "\n")

    def test_load_sessions(self):
        index = SessionIndex(str(self.session_dir))
        self.assertEqual(len(index.sessions), 5) # 3 valid, 1 empty, 1 malformed but still loaded
        filenames = {s["filename"] for s in index.sessions}
        self.assertIn("session_1.jsonl", filenames)
        self.assertIn("session_2.jsonl", filenames)
        self.assertIn("session_3.jsonl", filenames)
        self.assertIn("empty_session.jsonl", filenames)
        self.assertIn("malformed_session.jsonl", filenames)

    def test_extract_messages(self):
        index = SessionIndex(str(self.session_dir))
        session1_messages = next(s["messages"] for s in index.sessions if s["filename"] == "session_1.jsonl")
        self.assertEqual(len(session1_messages), 3)
        self.assertEqual(session1_messages[0], ("user", "Hello, how are you?"))
        self.assertEqual(session1_messages[1], ("assistant", "I'm fine, thank you."))

    def test_search_single_match(self):
        index = SessionIndex(str(self.session_dir))
        results = index.search("Python")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["session_file"], "session_2.jsonl")
        self.assertEqual(len(results[0]["matches"]), 2)

    def test_search_multiple_matches_same_session(self):
        index = SessionIndex(str(self.session_dir))
        results = index.search("you")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["session_file"], "session_1.jsonl")
        self.assertEqual(len(results[0]["matches"]), 3)

    def test_search_no_match(self):
        index = SessionIndex(str(self.session_dir))
        results = index.search("nonexistent")
        self.assertEqual(len(results), 0)

    def test_search_case_insensitivity(self):
        index = SessionIndex(str(self.session_dir))
        results = index.search("hello")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["session_file"], "session_1.jsonl")

    def test_search_max_results(self):
        # Create more sessions for max_results test
        self._create_session_file("session_4.jsonl", [
            {"user_message": "Another test session"}
        ])
        self._create_session_file("session_5.jsonl", [
            {"user_message": "Yet another test session"}
        ])

        index = SessionIndex(str(self.session_dir))
        results = index.search("session", max_results=2)
        self.assertEqual(len(results), 2)

    def test_search_with_malformed_lines(self):
        index = SessionIndex(str(self.session_dir))
        results = index.search("valid")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["session_file"], "malformed_session.jsonl")
        self.assertEqual(len(results[0]["matches"]), 2)
        self.assertEqual(results[0]["matches"][0]["message"], "Valid line")
        self.assertEqual(results[0]["matches"][1]["message"], "Another valid line")
