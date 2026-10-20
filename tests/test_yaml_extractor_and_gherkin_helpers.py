"""Tests for yaml_extractor and gherkin_extractor pure helpers."""


from mullet.knowledge.code.extractors.yaml_extractor import YAMLExtractor
from mullet.knowledge.code.extractors.gherkin_extractor import GherkinExtractor


# ---------------------------------------------------------------------------
# YAMLExtractor._detect_yaml_type
# ---------------------------------------------------------------------------

class TestDetectYamlType:
    def setup_method(self):
        self.e = YAMLExtractor()

    def test_docker_compose_by_services_and_version(self):
        data = {"services": {"web": {}}, "version": "3"}
        assert self.e._detect_yaml_type(data, "compose.yml") == "docker-compose"

    def test_docker_compose_by_filename(self):
        assert self.e._detect_yaml_type({}, "docker-compose.yml") == "docker-compose"

    def test_openapi_by_openapi_key(self):
        data = {"openapi": "3.0.0", "paths": {}}
        assert self.e._detect_yaml_type(data, "spec.yaml") == "openapi"

    def test_openapi_by_swagger_key(self):
        data = {"swagger": "2.0"}
        assert self.e._detect_yaml_type(data, "api.yaml") == "openapi"

    def test_openapi_by_paths_dict(self):
        data = {"paths": {"/users": {}}}
        assert self.e._detect_yaml_type(data, "unknown.yaml") == "openapi"

    def test_kubernetes_by_apiversion_and_kind(self):
        data = {"apiVersion": "v1", "kind": "Pod"}
        assert self.e._detect_yaml_type(data, "pod.yaml") == "kubernetes"

    def test_github_actions_by_on_and_jobs(self):
        data = {"on": "push", "jobs": {"build": {}}}
        assert self.e._detect_yaml_type(data, "build.yaml") == "github-actions"

    def test_github_actions_by_workflow_path(self):
        data = {"on": "push"}
        assert self.e._detect_yaml_type(data, ".github/workflows/ci.yaml") == "github-actions"

    def test_generic_fallback(self):
        data = {"key": "value"}
        assert self.e._detect_yaml_type(data, "config.yaml") == "generic"

    def test_returns_string(self):
        result = self.e._detect_yaml_type({}, "any.yaml")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# YAMLExtractor.extract
# ---------------------------------------------------------------------------

class TestYAMLExtractorExtract:
    def setup_method(self):
        self.e = YAMLExtractor()

    def test_language_is_yaml(self):
        assert self.e.language == "yaml"

    def test_empty_source_returns_chunk(self):
        result = self.e.extract("", "test.yaml")
        assert isinstance(result, list)

    def test_invalid_yaml_returns_chunk(self):
        result = self.e.extract("key: [\n unmatched", "bad.yaml")
        assert isinstance(result, list)

    def test_simple_yaml_returns_chunks(self):
        src = "name: myapp\nversion: 1.0\n"
        result = self.e.extract(src, "config.yaml")
        assert len(result) >= 1

    def test_kubernetes_yaml_extracted(self):
        src = "apiVersion: v1\nkind: Pod\nmetadata:\n  name: mypod\n"
        result = self.e.extract(src, "pod.yaml")
        assert len(result) >= 1

    def test_docker_compose_extracted(self):
        src = "version: '3'\nservices:\n  web:\n    image: nginx\n"
        result = self.e.extract(src, "docker-compose.yml")
        assert len(result) >= 1

    def test_chunks_have_source_file(self):
        src = "key: val\n"
        result = self.e.extract(src, "myfile.yaml")
        assert all(c.source_file == "myfile.yaml" for c in result)

    def test_openapi_yaml_extracted(self):
        src = "openapi: '3.0.0'\ninfo:\n  title: MyAPI\npaths:\n  /users:\n    get:\n      summary: List users\n"
        result = self.e.extract(src, "openapi.yaml")
        assert len(result) >= 1

    def test_returns_list(self):
        assert isinstance(self.e.extract("a: 1\n", "f.yaml"), list)


# ---------------------------------------------------------------------------
# GherkinExtractor.extract
# ---------------------------------------------------------------------------

class TestGherkinExtractorExtract:
    def setup_method(self):
        self.e = GherkinExtractor()

    def test_language_is_gherkin(self):
        assert self.e.language == "gherkin"

    def test_empty_source_returns_empty(self):
        result = self.e.extract("", "test.feature")
        assert isinstance(result, list)

    def test_feature_extracted(self):
        src = "Feature: Login\n  Scenario: Success\n    Given I am on the page\n"
        result = self.e.extract(src, "login.feature")
        features = [c for c in result if c.symbol_type == "feature"]
        assert len(features) == 1
        assert features[0].symbol_name == "Login"

    def test_scenario_extracted(self):
        src = "Feature: Login\n  Scenario: Success\n    Given I am on the page\n"
        result = self.e.extract(src, "login.feature")
        scenarios = [c for c in result if c.symbol_type == "scenario"]
        assert len(scenarios) == 1
        assert scenarios[0].symbol_name == "Success"

    def test_background_extracted(self):
        src = (
            "Feature: Login\n"
            "  Background:\n"
            "    Given the app is running\n"
            "  Scenario: S\n"
            "    Given step\n"
        )
        result = self.e.extract(src, "f.feature")
        bgs = [c for c in result if c.symbol_type == "background"]
        assert len(bgs) == 1

    def test_multiple_scenarios_extracted(self):
        src = (
            "Feature: Auth\n"
            "  Scenario: Login\n"
            "    Given I login\n"
            "  Scenario: Logout\n"
            "    Given I logout\n"
        )
        result = self.e.extract(src, "auth.feature")
        scenarios = [c for c in result if c.symbol_type == "scenario"]
        assert len(scenarios) == 2

    def test_chunk_source_file_set(self):
        src = "Feature: F\n  Scenario: S\n    Given step\n"
        result = self.e.extract(src, "my.feature")
        assert all(c.source_file == "my.feature" for c in result)

    def test_returns_list(self):
        assert isinstance(self.e.extract("Feature: F\n", "f.feature"), list)
