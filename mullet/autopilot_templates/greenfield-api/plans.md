# Project Plan — Greenfield API

## Architecture Overview
Layered: `api/` (routing + validation) -> `domain/` (business logic, pure) ->
`infra/` (db, external services). Dependencies flow inward: `domain/` imports nothing.

## Milestones

### Milestone 1: Skeleton
**Goal:** Running service with health check + one hello-world endpoint
**Depends on:** None

#### Tasks

##### Task 1.1: Project scaffolding
- **Parallel:** no
- **Files:** project config, package layout, lint/format setup
- **Approach:** generate from template, lock dependency versions
- **Tests:** import smoke test
- **Acceptance criteria:** `<build>` runs clean
- **Status:** pending

##### Task 1.2: Health + hello endpoint
- **Parallel:** no (depends on 1.1)
- **Files:** `api/health.*`, `api/hello.*`, tests
- **Approach:** simple router, return static JSON
- **Tests:** endpoint returns 200 + correct schema
- **Acceptance criteria:** `curl /health` returns 200
- **Status:** pending

### Milestone 2: Core domain
**Goal:** [primary resource] CRUD, persisted to DB
**Depends on:** Milestone 1

#### Tasks
##### Task 2.1: Domain model + repo interface
##### Task 2.2: SQL repo implementation
##### Task 2.3: CRUD endpoints
##### Task 2.4: Integration tests with real DB

### Milestone 3: Auth + observability
**Goal:** Production-grade cross-cutting concerns
**Depends on:** Milestone 2

#### Tasks
##### Task 3.1: Auth middleware (JWT / API key / whatever stack dictates)
##### Task 3.2: Structured logging + request-id propagation
##### Task 3.3: Metrics endpoint (Prometheus-compatible)
##### Task 3.4: OpenAPI doc generation

### Milestone 4: Deployability
**Goal:** CI green, container builds, runs in target environment
**Depends on:** Milestone 3

#### Tasks
##### Task 4.1: Dockerfile (multi-stage, non-root)
##### Task 4.2: CI pipeline (lint, test, build, publish)
##### Task 4.3: Deployment manifest (k8s / compose / serverless config)
##### Task 4.4: Smoke test against deployed instance
