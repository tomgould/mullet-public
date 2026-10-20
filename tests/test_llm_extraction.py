from mullet.knowledge.engine.llm_extraction import (
    CitationMap,
    build_citation_map,
    parse_citations_from_response,
    build_rag_prompt,
    categorize_chunks,
    build_rag_prompt_with_citations,
)

# Sample chunks for testing
sample_chunks = [
    {"content": "This is the first document.", "metadata": {"source_file": "doc1.txt", "chunk_type": "documentation"}},
    {"content": "def hello_world(): pass", "metadata": {"source_file": "code.py", "chunk_type": "code"}},
    {"content": "Another document here.", "metadata": {"source_file": "doc2.txt", "chunk_type": "documentation"}},
    {"content": "user: hi, how are you? agent: I am fine.", "metadata": {"source_file": "chat.log", "chunk_type": "conversation"}},
    {"content": "Some other content.", "metadata": {"source_file": "other.txt"}} # No chunk_type
]

def test_citation_map_creation():
    citation_map = CitationMap(chunks=sample_chunks)
    assert len(citation_map) == len(sample_chunks)

def test_citation_map_get_valid_index():
    citation_map = CitationMap(chunks=sample_chunks)
    assert citation_map.get(1) == sample_chunks[0]
    assert citation_map.get(2) == sample_chunks[1]

def test_citation_map_get_out_of_bounds_returns_none():
    citation_map = CitationMap(chunks=sample_chunks)
    assert citation_map.get(0) is None
    assert citation_map.get(len(sample_chunks) + 1) is None

def test_citation_map_len():
    citation_map = CitationMap(chunks=sample_chunks)
    assert len(citation_map) == 5
    empty_map = CitationMap(chunks=[])
    assert len(empty_map) == 0

def test_build_citation_map():
    citation_map = build_citation_map(sample_chunks)
    assert isinstance(citation_map, CitationMap)
    assert len(citation_map) == len(sample_chunks)
    assert citation_map.get(1) == sample_chunks[0]

def test_parse_citations_from_response_no_citations():
    response = "This is a response without any citations."
    assert parse_citations_from_response(response) == []

def test_parse_citations_from_response_single_citation():
    response = "Here is some info [1]."
    assert parse_citations_from_response(response) == [1]

def test_parse_citations_from_response_multiple_citations():
    response = "Info from [1] and [3] and also [2]."
    assert sorted(parse_citations_from_response(response)) == [1, 2, 3]

def test_parse_citations_from_response_duplicates():
    response = "Duplicate citations [1], [2], [1]."
    assert sorted(parse_citations_from_response(response)) == [1, 1, 2]

def test_build_rag_prompt_includes_chunks_and_question():
    question = "What is the capital of France?"
    prompt = build_rag_prompt(question, sample_chunks)
    assert question in prompt
    assert "This is the first document." in prompt
    assert "def hello_world(): pass" in prompt
    assert "[Source 1: doc1.txt]" in prompt
    assert "[Source 2: code.py]" in prompt

def test_build_rag_prompt_respects_max_context_tokens():
    question = "Short question?"
    # Set a very small max_context_tokens to ensure truncation
    prompt = build_rag_prompt(question, sample_chunks, max_context_tokens=50)
    # The exact content will depend on the _build_context_parts logic,
    # but it should definitely not contain all chunks.
    # We'll check for the first chunk and expect the last ones to be missing.
    assert "This is the first document." in prompt
    # The total chars for a 50 token prompt would be around 200.
    # The first chunk alone is already ~80 chars.
    # It's unlikely that both "doc2.txt" and "chat.log" will be in a short prompt.
    # We'll assert that the last chunk's content is NOT in the prompt.
    assert "Some other content." not in prompt

def test_categorize_chunks_code_vs_docs_vs_other():
    categories = categorize_chunks(sample_chunks)
    assert len(categories["code"]) == 1
    assert categories["code"][0]["content"] == "def hello_world(): pass"
    assert len(categories["documentation"]) == 2
    assert categories["documentation"][0]["content"] == "This is the first document."
    assert categories["documentation"][1]["content"] == "Another document here."
    assert len(categories["conversation"]) == 1
    assert categories["conversation"][0]["content"] == "user: hi, how are you? agent: I am fine."
    assert len(categories["other"]) == 1
    assert categories["other"][0]["content"] == "Some other content."

def test_categorize_chunks_empty_input():
    categories = categorize_chunks([])
    assert categories == {
        'code': [],
        'documentation': [],
        'conversation': [],
        'other': []
    }

def test_build_rag_prompt_with_citations_includes_citation_instruction():
    question = "How to do X?"
    prompt = build_rag_prompt_with_citations(question, sample_chunks)
    assert question in prompt
    assert "This is the first document." in prompt
    assert "def hello_world(): pass" in prompt
    assert "[1] doc1.txt" in prompt
    assert "[2] code.py" in prompt
    assert "Please provide a clear answer with citations to the sources, like [1], [2], etc." in prompt

def test_build_rag_prompt_with_citations_respects_max_context_tokens():
    question = "Short question with citations?"
    prompt = build_rag_prompt_with_citations(question, sample_chunks, max_context_tokens=50)
    assert "This is the first document." in prompt
    assert "Some other content." not in prompt