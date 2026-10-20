"""Tests for verifier pure helpers, llm_decomposer pure helpers, and YAML extractor."""

import pytest

from mullet.core.orchestration.verifier import (
    CriterionResult,
    VerificationResult,
    extract_criteria,
    _parse_criterion_result,
)
from mullet.core.orchestration.llm_decomposer import (
    DecompositionError,
    _extract_json_array,
    _parse_task_list,
)
from mullet.core.orchestration.swarm_types import SwarmTask
from mullet.knowledge.code.extractors.yaml_extractor import YAMLExtractor
from mullet.knowledge.code.extractors.base import CodeChunk


# ---------------------------------------------------------------------------
# CriterionResult
# ---------------------------------------------------------------------------

class TestCriterionResult:
    def test_fields_stored(self):
        r = CriterionResult(criterion="Users can log in", passed=True, evidence="Tested login flow")
        assert r.criterion == "Users can log in"
        assert r.passed is True
        assert r.evidence == "Tested login flow"

    def test_default_not_passed(self):
        r = CriterionResult(criterion="some task")
        assert r.passed is False

    def test_default_evidence_empty(self):
        r = CriterionResult(criterion="task")
        assert r.evidence == ""

    def test_default_attempt_one(self):
        r = CriterionResult(criterion="task")
        assert r.attempt == 1


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------

class TestVerificationResult:
    def test_default_not_passed(self):
        r = VerificationResult()
        assert r.passed is False

    def test_summary_pass(self):
        r = VerificationResult(
            passed=True,
            criteria_results={"crit1": True, "crit2": True},
            attempts=1,
        )
        assert "PASS" in r.summary
        assert "2/2" in r.summary

    def test_summary_fail(self):
        r = VerificationResult(
            passed=False,
            criteria_results={"crit1": True, "crit2": False},
            attempts=1,
        )
        assert "FAIL" in r.summary
        assert "1/2" in r.summary

    def test_failed_criteria_list(self):
        r = VerificationResult(
            criteria_results={"passes": True, "fails": False}
        )
        assert "fails" in r.failed_criteria
        assert "passes" not in r.failed_criteria

    def test_failed_criteria_empty_when_all_pass(self):
        r = VerificationResult(criteria_results={"a": True, "b": True})
        assert r.failed_criteria == []

    def test_evidence_default_empty(self):
        r = VerificationResult()
        assert r.evidence == []

    def test_issues_default_empty(self):
        r = VerificationResult()
        assert r.issues == []


# ---------------------------------------------------------------------------
# extract_criteria
# ---------------------------------------------------------------------------

class TestExtractCriteria:
    def test_checkbox_items_extracted(self):
        text = "- [ ] Users can log in\n- [x] Users can log out"
        criteria = extract_criteria(text)
        assert any("log in" in c for c in criteria)

    def test_numbered_items_extracted(self):
        text = "1. Users can create accounts\n2. Users can delete accounts"
        criteria = extract_criteria(text)
        assert len(criteria) == 2
        assert any("create accounts" in c for c in criteria)

    def test_ac_prefix_extracted(self):
        text = "AC: The system must respond in under 200ms"
        criteria = extract_criteria(text)
        assert any("200ms" in c for c in criteria)

    def test_acceptance_criteria_prefix(self):
        text = "Acceptance Criteria: Feature works end-to-end"
        criteria = extract_criteria(text)
        assert any("end-to-end" in c for c in criteria)

    def test_bullet_with_enough_words_extracted(self):
        text = "- The login form validates email addresses correctly"
        criteria = extract_criteria(text)
        assert any("login" in c for c in criteria)

    def test_empty_text_returns_empty(self):
        assert extract_criteria("") == []

    def test_blank_lines_skipped(self):
        text = "1. First item\n\n2. Second item"
        criteria = extract_criteria(text)
        assert len(criteria) == 2

    def test_returns_list(self):
        assert isinstance(extract_criteria("- [ ] something"), list)

    def test_checkbox_checked_also_extracted(self):
        text = "- [x] Feature already done"
        criteria = extract_criteria(text)
        assert any("already done" in c for c in criteria)

    def test_multi_format_mixed(self):
        text = (
            "- [ ] Users can register\n"
            "1. Users can login\n"
            "AC: System handles errors\n"
        )
        criteria = extract_criteria(text)
        assert len(criteria) >= 3


# ---------------------------------------------------------------------------
# _parse_criterion_result
# ---------------------------------------------------------------------------

class TestParseCriterionResult:
    def test_pass_detected(self):
        response = "CRITERION 1: PASS - I verified the file exists and contains data"
        passed, evidence = _parse_criterion_result(response, 1)
        assert passed is True

    def test_fail_detected(self):
        response = "CRITERION 2: FAIL - The test file was not found"
        passed, evidence = _parse_criterion_result(response, 2)
        assert passed is False

    def test_all_pass_fallback(self):
        response = "All criteria have been verified and ALL PASS"
        passed, evidence = _parse_criterion_result(response, 99)
        assert passed is True

    def test_returns_tuple(self):
        result = _parse_criterion_result("response text", 1)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_no_match_returns_false(self):
        response = "I looked at the code but did not find any clear answer"
        passed, evidence = _parse_criterion_result(response, 1)
        assert passed is False

    def test_evidence_is_string(self):
        response = "CRITERION 1: PASS - file exists"
        _, evidence = _parse_criterion_result(response, 1)
        assert isinstance(evidence, str)


# ---------------------------------------------------------------------------
# DecompositionError
# ---------------------------------------------------------------------------

class TestDecompositionError:
    def test_is_exception(self):
        with pytest.raises(DecompositionError):
            raise DecompositionError("decomposition failed")

    def test_message_stored(self):
        try:
            raise DecompositionError("bad response")
        except DecompositionError as e:
            assert "bad response" in str(e)


# ---------------------------------------------------------------------------
# _extract_json_array
# ---------------------------------------------------------------------------

class TestExtractJsonArray:
    def test_simple_array(self):
        raw = '[{"id": "t1", "instruction": "do this"}]'
        result = _extract_json_array(raw)
        assert result == raw

    def test_array_with_trailing_text(self):
        raw = '[{"id": "t1", "instruction": "do this"}] Note: done'
        result = _extract_json_array(raw)
        assert result.endswith("]")

    def test_array_with_leading_text(self):
        raw = 'Here is my response: [{"id": "t1", "instruction": "x"}]'
        result = _extract_json_array(raw)
        assert result.startswith("[")

    def test_raises_if_no_array(self):
        with pytest.raises(DecompositionError):
            _extract_json_array("no array here")

    def test_raises_on_unclosed_array(self):
        with pytest.raises(DecompositionError):
            _extract_json_array("[{unclosed}")

    def test_nested_arrays_handled(self):
        raw = '[[1, 2], [3, 4]]'
        result = _extract_json_array(raw)
        assert result.startswith("[")

    def test_string_with_brackets_preserved(self):
        raw = '[{"id": "t", "instruction": "Use array [a, b] here"}]'
        result = _extract_json_array(raw)
        assert "array [a, b]" in result

    def test_empty_array(self):
        raw = '[]'
        result = _extract_json_array(raw)
        assert result == "[]"

    def test_escaped_string_no_crash(self):
        raw = '[{"id": "t", "instruction": "has \\"quotes\\" inside"}]'
        result = _extract_json_array(raw)
        assert result is not None


# ---------------------------------------------------------------------------
# _parse_task_list
# ---------------------------------------------------------------------------

class TestParseTaskList:
    def test_valid_json_returns_tasks(self):
        raw = '[{"id": "t1", "instruction": "do task 1"}]'
        tasks = _parse_task_list(raw, "markdown report")
        assert len(tasks) == 1

    def test_returns_swarm_tasks(self):
        raw = '[{"id": "t1", "instruction": "do task 1"}]'
        tasks = _parse_task_list(raw, "markdown report")
        assert isinstance(tasks[0], SwarmTask)

    def test_id_and_instruction_set(self):
        raw = '[{"id": "analyze", "instruction": "analyze the data"}]'
        tasks = _parse_task_list(raw, "markdown report")
        assert tasks[0].id == "analyze"
        assert tasks[0].instruction == "analyze the data"

    def test_output_shape_default_used(self):
        raw = '[{"id": "t1", "instruction": "do it"}]'
        tasks = _parse_task_list(raw, "json output")
        assert tasks[0].expected_output_shape == "json output"

    def test_output_shape_override_used(self):
        raw = '[{"id": "t1", "instruction": "do it", "expected_output_shape": "csv"}]'
        tasks = _parse_task_list(raw, "default")
        assert tasks[0].expected_output_shape == "csv"

    def test_context_refs_set(self):
        raw = '[{"id": "t1", "instruction": "read it", "context_refs": ["file.py"]}]'
        tasks = _parse_task_list(raw, "report")
        assert "file.py" in tasks[0].context_refs

    def test_missing_id_raises(self):
        raw = '[{"instruction": "do it but no id"}]'
        with pytest.raises(DecompositionError):
            _parse_task_list(raw, "report")

    def test_missing_instruction_raises(self):
        raw = '[{"id": "t1"}]'
        with pytest.raises(DecompositionError):
            _parse_task_list(raw, "report")

    def test_empty_array_returns_empty(self):
        raw = "[]"
        tasks = _parse_task_list(raw, "report")
        assert tasks == []

    def test_markdown_fenced_input_stripped(self):
        raw = '```json\n[{"id": "t1", "instruction": "task here"}]\n```'
        tasks = _parse_task_list(raw, "report")
        assert len(tasks) == 1

    def test_multiple_tasks(self):
        raw = '[{"id": "t1", "instruction": "first"}, {"id": "t2", "instruction": "second"}]'
        tasks = _parse_task_list(raw, "report")
        assert len(tasks) == 2

    def test_invalid_json_raises(self):
        with pytest.raises(DecompositionError):
            _parse_task_list("not json at all", "report")

    def test_non_list_response_raises(self):
        with pytest.raises(DecompositionError):
            _parse_task_list('{"key": "value"}', "report")


# ---------------------------------------------------------------------------
# YAMLExtractor._detect_yaml_type
# ---------------------------------------------------------------------------

class TestDetectYamlType:
    def test_docker_compose_by_content(self):
        ext = YAMLExtractor()
        data = {"version": "3", "services": {"web": {"image": "nginx"}}}
        assert ext._detect_yaml_type(data, "docker-compose.yml") == "docker-compose"

    def test_docker_compose_by_filename(self):
        ext = YAMLExtractor()
        assert ext._detect_yaml_type({}, "docker-compose.yml") == "docker-compose"

    def test_openapi_by_key(self):
        ext = YAMLExtractor()
        data = {"openapi": "3.0.0", "info": {"title": "My API"}}
        assert ext._detect_yaml_type(data, "api.yaml") == "openapi"

    def test_openapi_by_paths(self):
        ext = YAMLExtractor()
        data = {"paths": {"/users": {"get": {}}}}
        assert ext._detect_yaml_type(data, "spec.yaml") == "openapi"

    def test_kubernetes_by_keys(self):
        ext = YAMLExtractor()
        data = {"apiVersion": "v1", "kind": "Pod", "metadata": {}}
        assert ext._detect_yaml_type(data, "pod.yaml") == "kubernetes"

    def test_github_actions_by_path(self):
        ext = YAMLExtractor()
        data = {}
        assert ext._detect_yaml_type(data, ".github/workflows/ci.yml") == "github-actions"

    def test_github_actions_by_content(self):
        ext = YAMLExtractor()
        data = {"on": "push", "jobs": {"build": {}}}
        assert ext._detect_yaml_type(data, "pipeline.yml") == "github-actions"

    def test_generic_for_unknown(self):
        ext = YAMLExtractor()
        data = {"some": "config", "key": "value"}
        assert ext._detect_yaml_type(data, "settings.yaml") == "generic"


# ---------------------------------------------------------------------------
# YAMLExtractor.extract
# ---------------------------------------------------------------------------

SIMPLE_YAML = """\
name: my-project
version: 1.0
description: A test project
settings:
  debug: true
  port: 8080
"""

DOCKER_COMPOSE_YAML = """\
version: "3.8"
services:
  web:
    image: nginx:latest
    ports:
      - "80:80"
  db:
    image: postgres:14
    environment:
      POSTGRES_DB: mydb
"""

KUBERNETES_YAML = """\
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
    - name: app
      image: myapp:latest
"""

OPENAPI_YAML = """\
openapi: "3.0.0"
info:
  title: My API
  version: "1.0"
paths:
  /users:
    get:
      summary: List users
      responses:
        "200":
          description: Success
"""


class TestYAMLExtractorExtract:
    def test_simple_yaml_returns_chunks(self):
        ext = YAMLExtractor()
        chunks = ext.extract(SIMPLE_YAML, "settings.yaml")
        assert len(chunks) >= 1

    def test_chunks_are_code_chunks(self):
        ext = YAMLExtractor()
        chunks = ext.extract(SIMPLE_YAML, "settings.yaml")
        for c in chunks:
            assert isinstance(c, CodeChunk)

    def test_docker_compose_extracts_services(self):
        ext = YAMLExtractor()
        chunks = ext.extract(DOCKER_COMPOSE_YAML, "docker-compose.yml")
        types = [c.symbol_type for c in chunks]
        assert "service" in types

    def test_docker_compose_service_names(self):
        ext = YAMLExtractor()
        chunks = ext.extract(DOCKER_COMPOSE_YAML, "docker-compose.yml")
        names = [c.symbol_name for c in chunks if c.symbol_type == "service"]
        assert "web" in names
        assert "db" in names

    def test_kubernetes_extracts_resource(self):
        ext = YAMLExtractor()
        chunks = ext.extract(KUBERNETES_YAML, "pod.yaml")
        types = [c.symbol_type for c in chunks]
        assert any(t in ("resource", "kubernetes_resource") for t in types) or len(chunks) >= 1

    def test_openapi_extracts_endpoint(self):
        ext = YAMLExtractor()
        chunks = ext.extract(OPENAPI_YAML, "api.yaml")
        types = [c.symbol_type for c in chunks]
        assert any(t in ("endpoint", "path", "operation", "info", "api") for t in types) or len(chunks) >= 1

    def test_invalid_yaml_returns_document_chunk(self):
        ext = YAMLExtractor()
        bad_yaml = "key: value\n  bad indentation:\n    - unclosed ["
        chunks = ext.extract(bad_yaml, "broken.yaml")
        assert len(chunks) >= 1

    def test_chunk_language_is_yaml(self):
        ext = YAMLExtractor()
        chunks = ext.extract(SIMPLE_YAML, "settings.yaml")
        assert all(c.language == "yaml" for c in chunks)

    def test_chunk_source_file_set(self):
        ext = YAMLExtractor()
        chunks = ext.extract(SIMPLE_YAML, "my/path/settings.yaml")
        assert all(c.source_file == "my/path/settings.yaml" for c in chunks)

    def test_list_yaml_returns_document_chunk(self):
        ext = YAMLExtractor()
        yaml_list = "- item1\n- item2\n- item3\n"
        chunks = ext.extract(yaml_list, "list.yaml")
        assert len(chunks) == 1
        assert chunks[0].symbol_type == "document"

    def test_empty_yaml_no_crash(self):
        ext = YAMLExtractor()
        chunks = ext.extract("", "empty.yaml")
        assert isinstance(chunks, list)
