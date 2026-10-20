"""Tests for mullet.knowledge.code.extractors.yaml_extractor — pure helpers."""

import yaml
from mullet.knowledge.code.extractors.yaml_extractor import (
    YAMLExtractor,
    _PermissiveYAMLLoader,
)

EXTRACTOR = YAMLExtractor()


# ---------------------------------------------------------------------------
# _detect_yaml_type
# ---------------------------------------------------------------------------

class TestDetectYamlType:
    def test_docker_compose_by_filename(self):
        assert EXTRACTOR._detect_yaml_type({}, "docker-compose.yml") == "docker-compose"

    def test_compose_in_filename(self):
        assert EXTRACTOR._detect_yaml_type({}, "compose.yaml") == "docker-compose"

    def test_docker_compose_by_content_services_and_version(self):
        data = {"services": {"web": {}}, "version": "3"}
        assert EXTRACTOR._detect_yaml_type(data, "any.yml") == "docker-compose"

    def test_openapi_by_openapi_key(self):
        data = {"openapi": "3.0.0", "info": {}}
        assert EXTRACTOR._detect_yaml_type(data, "spec.yaml") == "openapi"

    def test_openapi_by_swagger_key(self):
        data = {"swagger": "2.0", "info": {}}
        assert EXTRACTOR._detect_yaml_type(data, "spec.yaml") == "openapi"

    def test_openapi_by_paths(self):
        data = {"paths": {"/users": {}}}
        assert EXTRACTOR._detect_yaml_type(data, "spec.yaml") == "openapi"

    def test_kubernetes_by_apiversion_and_kind(self):
        data = {"apiVersion": "v1", "kind": "Pod", "metadata": {}}
        assert EXTRACTOR._detect_yaml_type(data, "deployment.yaml") == "kubernetes"

    def test_github_actions_by_filepath(self):
        data = {"on": "push"}
        assert EXTRACTOR._detect_yaml_type(data, ".github/workflows/ci.yml") == "github-actions"

    def test_github_actions_by_content_on_jobs(self):
        data = {"on": {"push": {}}, "jobs": {"build": {}}}
        assert EXTRACTOR._detect_yaml_type(data, "ci.yml") == "github-actions"

    def test_generic_fallback(self):
        data = {"foo": "bar", "baz": 42}
        assert EXTRACTOR._detect_yaml_type(data, "config.yaml") == "generic"


# ---------------------------------------------------------------------------
# _PermissiveYAMLLoader — custom tag handling
# ---------------------------------------------------------------------------

class TestPermissiveLoader:
    def test_unknown_tag_not_raises(self):
        source = "resource: !GetAtt MyResource.Arn"
        result = yaml.load(source, Loader=_PermissiveYAMLLoader)
        assert result is not None

    def test_scalar_tag_becomes_string(self):
        source = "ref: !Ref MyBucket"
        result = yaml.load(source, Loader=_PermissiveYAMLLoader)
        assert isinstance(result["ref"], str)

    def test_plain_yaml_still_works(self):
        source = "key: value\nnum: 42"
        result = yaml.load(source, Loader=_PermissiveYAMLLoader)
        assert result["key"] == "value"
        assert result["num"] == 42


# ---------------------------------------------------------------------------
# YAMLExtractor.extract — integration
# ---------------------------------------------------------------------------

class TestYAMLExtractorExtract:
    def test_returns_list(self):
        result = EXTRACTOR.extract("key: value\n", "config.yaml")
        assert isinstance(result, list)

    def test_invalid_yaml_returns_fallback(self):
        result = EXTRACTOR.extract(":: invalid :: yaml ::", "bad.yaml")
        assert isinstance(result, list)

    def test_docker_compose_extracts_services(self):
        source = yaml.dump({
            "version": "3",
            "services": {
                "web": {"image": "nginx", "ports": ["80:80"]},
                "db": {"image": "postgres"},
            }
        })
        chunks = EXTRACTOR.extract(source, "docker-compose.yml")
        service_names = {c.symbol_name for c in chunks if c.symbol_type == "service"}
        assert "web" in service_names
        assert "db" in service_names

    def test_kubernetes_manifest_extracts_chunk(self):
        source = yaml.dump({
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "my-config"},
            "data": {"key": "value"},
        })
        chunks = EXTRACTOR.extract(source, "configmap.yaml")
        assert any(c.symbol_type == "resource" for c in chunks)

    def test_github_actions_extracts_workflow(self):
        source = yaml.dump({
            "name": "CI",
            "on": {"push": {"branches": ["main"]}},
            "jobs": {"build": {"runs-on": "ubuntu-latest", "steps": []}},
        })
        chunks = EXTRACTOR.extract(source, ".github/workflows/ci.yml")
        assert len(chunks) > 0

    def test_openapi_extracts_paths(self):
        source = yaml.dump({
            "openapi": "3.0.0",
            "info": {"title": "My API", "version": "1.0"},
            "paths": {
                "/users": {"get": {"summary": "List users"}},
            }
        })
        chunks = EXTRACTOR.extract(source, "openapi.yaml")
        assert len(chunks) > 0

    def test_generic_yaml_chunk_has_source_file(self):
        source = "name: test\nversion: 1\n"
        chunks = EXTRACTOR.extract(source, "config.yaml")
        assert all(c.source_file == "config.yaml" for c in chunks)

    def test_null_yaml_returns_fallback(self):
        chunks = EXTRACTOR.extract("null\n", "null.yaml")
        assert isinstance(chunks, list)
