"""Tree-sitter backed Python code extractor that emits CodeChunk records."""
from typing import List, Optional
from tree_sitter_languages import get_language, get_parser

class CodeChunk:
    """Represents an extracted chunk of code."""

    def __init__(self, name: str, type: str, content: str, start_line: int, end_line: int):
        self.name = name
        self.type = type
        self.content = content
        self.start_line = start_line
        self.end_line = end_line

    def __repr__(self):
        return f"CodeChunk(name='{self.name}', type='{self.type}', start_line={self.start_line}, end_line={self.end_line})"

class PythonCodeExtractor:
    """Extract functions, classes, and imports from Python source."""

    def __init__(self):
        self.parser = get_parser("python")
        self.language = get_language("python")

    def _get_node_text(self, node, source_bytes: bytes) -> str:
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def _extract_function_or_class(self, node, source_bytes: bytes, type_name: str) -> Optional[CodeChunk]:
        name_node = None
        if type_name == "function" or type_name == "class":
            name_node = node.child_by_field_name("name")

        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)
        content = self._get_node_text(node, source_bytes)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        return CodeChunk(name, type_name, content, start_line, end_line)

    def _extract_import(self, node, source_bytes: bytes) -> Optional[CodeChunk]:
        content = self._get_node_text(node, source_bytes)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        return CodeChunk(f"import_{start_line}", "import", content, start_line, end_line)

    def extract(self, source_code: str) -> List[CodeChunk]:
        """Extract code chunks from the given source code."""
        chunks = []
        source_bytes = source_code.encode("utf-8")
        tree = self.parser.parse(source_bytes)
        root_node = tree.root_node

        for node in root_node.children:
            if node.type == "function_definition":
                chunk = self._extract_function_or_class(node, source_bytes, "function")
                if chunk:
                    chunks.append(chunk)
            elif node.type == "class_definition":
                chunk = self._extract_function_or_class(node, source_bytes, "class")
                if chunk:
                    chunks.append(chunk)
            elif node.type in ["import_statement", "import_from_statement"]:
                chunk = self._extract_import(node, source_bytes)
                if chunk:
                    chunks.append(chunk)
        return chunks

def register():
    """Register the Python code extractor."""
    return {
        "python": PythonCodeExtractor
    }
