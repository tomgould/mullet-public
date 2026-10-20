from mullet.core.tools.fuzzy import fuzzy_match

def test_fuzzy_match_exact():
    candidates = ["apple", "banana", "orange"]
    result = fuzzy_match("apple", candidates)
    assert len(result) == 1
    assert result[0][0] == "apple"
    assert result[0][1] == 1.0

def test_fuzzy_match_partial():
    candidates = ["apple pie", "banana split", "orange juice"]
    result = fuzzy_match("apple", candidates)
    assert len(result) == 1
    assert result[0][0] == "apple pie"
    assert result[0][1] > 0.6 # should be around 0.76

def test_fuzzy_match_no_match():
    candidates = ["grape", "kiwi", "pear"]
    result = fuzzy_match("apple", candidates)
    assert len(result) == 0

def test_fuzzy_match_case_insensitivity():
    candidates = ["Apple", "BANANA", "Orange"]
    result = fuzzy_match("apple", candidates)
    assert len(result) == 1
    assert result[0][0] == "Apple"
    assert result[0][1] == 1.0

def test_fuzzy_match_threshold():
    candidates = ["apple", "aple", "apply"]
    result = fuzzy_match("apple", candidates, threshold=0.9)
    assert len(result) == 1
    assert result[0][0] == "apple"

def test_fuzzy_match_multiple_matches():
    candidates = ["apple", "apply", "apps", "banana"]
    result = fuzzy_match("appl", candidates, threshold=0.5)
    assert len(result) == 3
    assert result[0][0] == "apple"
    assert result[1][0] == "apply"
    assert result[2][0] == "apps"

def test_fuzzy_match_empty_candidates():
    candidates = []
    result = fuzzy_match("apple", candidates)
    assert len(result) == 0

def test_fuzzy_match_empty_query():
    candidates = ["apple", "banana"]
    result = fuzzy_match("", candidates)
    assert len(result) == 0 # SequenceMatcher returns 0.0 for empty strings
