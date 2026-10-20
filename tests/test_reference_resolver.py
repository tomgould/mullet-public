from mullet.core.knowledge.reference_resolver import detect_references, resolve_reference, Reference

# Mock document index for resolve_reference tests
MOCK_DOCUMENT_INDEX = {
    "doc1.txt": "This is Section 1.1 of doc1. For more, see Appendix A. Also, Table 2.1 is important.",
    "doc2.txt": "Refer to Section 3.2 for details. Figure 1.5 shows the data.",
    "legal_doc.txt": "Article 5.1 outlines the terms. See Exhibit B.",
    "academic_doc.txt": "Chapter 2 discusses methods. Table 3.4 provides results. Equation 1.1 is key."
}

def test_detect_tech_doc_references():
    text = "See Section 3.1, Appendix A, Table 2.1, and Figure 1.2."
    references = detect_references(text, doc_type="tech_docs")
    assert len(references) == 4
    assert any(r.type == "Section" and r.identifier == "3.1" for r in references)
    assert any(r.type == "Appendix" and r.identifier == "A" for r in references)
    assert any(r.type == "Table" and r.identifier == "2.1" for r in references)
    print(f"Detected references: {references}")
    assert any(r.type == "Section" and r.identifier == "3.1" for r in references)
    assert any(r.type == "Appendix" and r.identifier == "A" for r in references)
    assert any(r.type == "Table" and r.identifier == "2.1" for r in references)
    assert any(r.type == "Figure" and r.identifier == "1.2" for r in references)

def test_detect_legal_doc_references():
    text = "According to Article 5.1 and Paragraph 2.3. Also, consult Section 10.1 and Exhibit B."
    references = detect_references(text, doc_type="legal_docs")
    print(f"Detected references: {references}")
    assert len(references) == 4
    assert any(r.type == "Article" and r.identifier == "5.1" for r in references)
    assert any(r.type == "Paragraph" and r.identifier == "2.3" for r in references)
    assert any(r.type == "Section" and r.identifier == "10.1" for r in references)
    assert any(r.type == "Exhibit" and r.identifier == "B" for r in references)

def test_detect_academic_doc_references():
    text = "As per Chapter 4, Section 2.2, Appendix C, Table 1.1, Figure 3.4 and Equation 5.0."
    references = detect_references(text, doc_type="academic_docs")
    print(f"Detected references: {references}")
    assert len(references) == 6
    assert any(r.type == "Chapter" and r.identifier == "4" for r in references)
    assert any(r.type == "Section" and r.identifier == "2.2" for r in references)
    assert any(r.type == "Appendix" and r.identifier == "C" for r in references)
    assert any(r.type == "Table" and r.identifier == "1.1" for r in references)
    assert any(r.type == "Figure" and r.identifier == "3.4" for r in references)
    assert any(r.type == "Equation" and r.identifier == "5.0" for r in references)

def test_detect_references_empty_text():
    text = ""
    references = detect_references(text)
    assert len(references) == 0

def test_detect_references_no_matches():
    text = "This text has no references."
    references = detect_references(text)
    assert len(references) == 0

def test_resolve_reference_found():
    ref = Reference(type="Section", identifier="1.1", position=0)
    resolved = resolve_reference(ref, MOCK_DOCUMENT_INDEX)
    assert resolved is not None
    assert resolved["document_id"] == "doc1.txt"
    assert resolved["match_text"] == "Section 1.1"

def test_resolve_reference_not_found():
    ref = Reference(type="Section", identifier="99.9", position=0)
    resolved = resolve_reference(ref, MOCK_DOCUMENT_INDEX)
    assert resolved is None

def test_resolve_reference_different_type():
    ref = Reference(type="Figure", identifier="1.5", position=0)
    resolved = resolve_reference(ref, MOCK_DOCUMENT_INDEX)
    assert resolved is not None
    assert resolved["document_id"] == "doc2.txt"
    assert resolved["match_text"] == "Figure 1.5"

def test_resolve_reference_case_insensitivity():
    ref = Reference(type="appendix", identifier="A", position=0) # Lowercase type
    resolved = resolve_reference(ref, MOCK_DOCUMENT_INDEX)
    assert resolved is not None
    assert resolved["document_id"] == "doc1.txt"
    assert resolved["match_text"] == "Appendix A"
