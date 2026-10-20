"""Tests for mullet.knowledge.code.extractors.yaml_extractor."""
from __future__ import annotations

import pytest

from mullet.knowledge.code.extractors.yaml_extractor import YAMLExtractor


_DOCKER_COMPOSE = """\
version: "3.8"
services:
  web:
    image: nginx:latest
    ports:
      - "80:80"
    depends_on:
      - db
  db:
    image: postgres:15
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
"""

_OPENAPI = """\
openapi: "3.0.0"
info:
  title: Test API
  version: "1.0"
paths:
  /users:
    get:
      summary: List users
      operationId: listUsers
      tags:
        - users
  /users/{id}:
    get:
      summary: Get user by ID
      operationId: getUser
"""

_KUBERNETES = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  namespace: production
spec:
  replicas: 3
"""

_GITHUB_ACTIONS = """\
name: CI
on:
  push:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v3
      - name: Run tests
        run: pytest
  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - name: Build
        run: docker build .
"""

_GENERIC = """\
database:
  host: localhost
  port: 5432
cache:
  backend: redis
  timeout: 300
"""


@pytest.fixture
def extractor() -> YAMLExtractor:
    return YAMLExtractor()


class TestDockerComposeExtraction:
    def test_extracts_services(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract(_DOCKER_COMPOSE, "docker-compose.yml")
        service_names = [c.symbol_name for c in chunks if c.symbol_type == "service"]
        assert "web" in service_names
        assert "db" in service_names

    def test_service_symbol_type(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract(_DOCKER_COMPOSE, "docker-compose.yml")
        types = {c.symbol_type for c in chunks}
        assert "service" in types

    def test_service_image_in_extras(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract(_DOCKER_COMPOSE, "docker-compose.yml")
        web = next(c for c in chunks if c.symbol_name == "web")
        assert web.extras.get("image") == "nginx:latest"

    def test_service_ports_in_extras(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract(_DOCKER_COMPOSE, "docker-compose.yml")
        web = next(c for c in chunks if c.symbol_name == "web")
        assert "80:80" in web.extras.get("ports", [])

    def test_service_language_is_yaml(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract(_DOCKER_COMPOSE, "docker-compose.yml")
        assert all(c.language == "yaml" for c in chunks)

    def test_source_file_set(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract(_DOCKER_COMPOSE, "docker-compose.yml")
        assert all(c.source_file == "docker-compose.yml" for c in chunks)


class TestOpenAPIExtraction:
    def test_extracts_endpoints(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract(_OPENAPI, "openapi.yaml")
        endpoint_types = [c.symbol_type for c in chunks if c.symbol_type == "endpoint"]
        assert len(endpoint_types) >= 2

    def test_endpoint_names_include_method_and_path(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract(_OPENAPI, "openapi.yaml")
        names = [c.symbol_name for c in chunks if c.symbol_type == "endpoint"]
        assert any("GET" in n and "/users" in n for n in names)

    def test_operation_id_in_extras(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract(_OPENAPI, "openapi.yaml")
        endpoints = [c for c in chunks if c.symbol_type == "endpoint"]
        op_ids = [c.extras.get("operation_id") for c in endpoints]
        assert "listUsers" in op_ids or "getUser" in op_ids


class TestKubernetesExtraction:
    def test_extracts_resource(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract(_KUBERNETES, "deployment.yaml")
        resources = [c for c in chunks if c.symbol_type == "resource"]
        assert len(resources) == 1

    def test_resource_name_includes_kind(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract(_KUBERNETES, "deployment.yaml")
        resource = chunks[0]
        assert "Deployment" in resource.symbol_name
        assert "my-app" in resource.symbol_name

    def test_kubernetes_extras(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract(_KUBERNETES, "deployment.yaml")
        resource = chunks[0]
        assert resource.extras.get("kind") == "Deployment"
        assert resource.extras.get("name") == "my-app"
        assert resource.extras.get("namespace") == "production"


class TestGitHubActionsExtraction:
    def test_extracts_jobs(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract(_GITHUB_ACTIONS, ".github/workflows/ci.yml")
        job_chunks = [c for c in chunks if c.symbol_type == "job"]
        assert len(job_chunks) == 2

    def test_job_names(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract(_GITHUB_ACTIONS, ".github/workflows/ci.yml")
        job_names = [c.symbol_name for c in chunks if c.symbol_type == "job"]
        assert "test" in job_names
        assert "build" in job_names

    def test_job_extras_include_runs_on(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract(_GITHUB_ACTIONS, ".github/workflows/ci.yml")
        test_job = next(c for c in chunks if c.symbol_name == "test")
        assert test_job.extras.get("runs_on") == "ubuntu-latest"

    def test_job_extras_include_needs(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract(_GITHUB_ACTIONS, ".github/workflows/ci.yml")
        build_job = next(c for c in chunks if c.symbol_name == "build")
        assert "test" in build_job.extras.get("needs", [])


class TestGenericYAMLExtraction:
    def test_extracts_top_level_keys(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract(_GENERIC, "config.yaml")
        key_names = [c.symbol_name for c in chunks if c.symbol_type == "key"]
        assert "database" in key_names
        assert "cache" in key_names

    def test_generic_chunk_has_yaml_type(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract(_GENERIC, "config.yaml")
        assert all(c.extras.get("yaml_type") == "generic" for c in chunks)


class TestEdgeCases:
    def test_empty_yaml_returns_list(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract("", "empty.yaml")
        assert isinstance(chunks, list)

    def test_null_yaml_returns_document_chunk(self, extractor: YAMLExtractor) -> None:
        chunks = extractor.extract("null", "null.yaml")
        assert len(chunks) == 1
        assert chunks[0].symbol_type == "document"

    def test_invalid_yaml_returns_document_chunk(self, extractor: YAMLExtractor) -> None:
        # Deeply invalid YAML that can't be parsed at all
        chunks = extractor.extract("{bad yaml: [unclosed", "bad.yaml")
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_large_content_truncated(self, extractor: YAMLExtractor) -> None:
        big_yaml = "key: '" + "x" * 3000 + "'"
        chunks = extractor.extract(big_yaml, "big.yaml")
        assert any(len(c.content) <= 2100 for c in chunks)

    def test_cloudformation_tag_handled(self, extractor: YAMLExtractor) -> None:
        cfn = """\
AWSTemplateFormatVersion: "2010-09-09"
Resources:
  MyBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub "${AWS::StackName}-bucket"
"""
        chunks = extractor.extract(cfn, "template.yaml")
        assert isinstance(chunks, list)
        assert len(chunks) > 0
