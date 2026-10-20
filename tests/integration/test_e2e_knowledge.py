"""End-to-end knowledge pipeline test — no real LLM calls."""

import pytest
from mullet.knowledge.code import CodeParser, CodeChunker
from mullet.knowledge.graph.models import Entity
from mullet.knowledge.graph.store import KnowledgeGraph
from mullet.knowledge.engine.llm_extraction import build_rag_prompt, build_citation_map


@pytest.fixture
def code_dir(tmp_path):
    (tmp_path / "module1.py").write_text(
        'def greet(name: str) -> str:\n    """Greets the given name."""\n    return f"Hello, {name}!"\n\n'
        'class Calculator:\n    """A simple calculator."""\n    def add(self, a, b):\n        return a + b\n'
    )
    (tmp_path / "module2.py").write_text(
        'def subtract(a: int, b: int) -> int:\n    """Subtracts b from a."""\n    return a - b\n\n'
        'MAX_VALUE = 9999\n'
    )
    return tmp_path


def test_code_chunking(code_dir):
    parser = CodeParser()
    chunks1 = parser.parse_file(code_dir / "module1.py")
    chunks2 = parser.parse_file(code_dir / "module2.py")
    all_chunks = chunks1 + chunks2
    assert len(all_chunks) >= 4  # greet, Calculator, add, subtract, maybe MAX_VALUE
    names = {c.symbol_name for c in all_chunks}
    assert "greet" in names
    assert "Calculator" in names
    assert "subtract" in names


def test_code_chunker_directory(code_dir):
    chunker = CodeChunker()
    chunks = chunker.chunk_paths([code_dir])
    assert len(chunks) >= 4
    # chunks are dicts from to_dict()
    symbols = {c["metadata"]["symbol_name"] for c in chunks}
    assert "greet" in symbols


def test_knowledge_graph_from_chunks(code_dir, tmp_path):
    parser = CodeParser()
    chunks = parser.parse_file(code_dir / "module1.py")
    
    kg = KnowledgeGraph(tmp_path / "graph")
    for chunk in chunks:
        entity = Entity(
            name=chunk.symbol_name,
            entity_type=chunk.symbol_type,
            description=chunk.docstring or "",
            source_chunk_ids=[f"{chunk.source_file}:{chunk.line_start}"],
        )
        kg.add_entity(entity)
    
    assert kg.stats["entity_count"] >= 2
    found = kg.find_entities_by_name("greet")
    assert len(found) >= 1
    kg.close()


def test_rag_prompt_from_chunks(code_dir):
    chunker = CodeChunker()
    chunks = chunker.chunk_paths([code_dir])
    
    prompt = build_rag_prompt("How does greeting work?", chunks)
    assert "How does greeting work?" in prompt
    assert "greet" in prompt


def test_citation_map(code_dir):
    chunker = CodeChunker()
    chunks = chunker.chunk_paths([code_dir])
    cm = build_citation_map(chunks)
    assert len(cm) >= 4
    assert cm.get(1) is not None
    assert cm.get(0) is None  # 1-indexed
