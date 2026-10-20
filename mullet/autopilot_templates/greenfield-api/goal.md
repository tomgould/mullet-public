# Project Goal — Greenfield API

## Problem Statement
[Which domain capability is missing? Who's the consumer? Why now?]

## Desired Outcome
A production-ready API service with:
- Clear resource model and versioned endpoints
- Auth + authorisation story decided from day 1
- Observability (logs + metrics + traces) baked in
- Deployable via container image

## Acceptance Criteria
- [ ] OpenAPI spec published
- [ ] ≥85% unit test coverage on domain logic
- [ ] Integration tests cover every endpoint's happy + 4xx + 5xx paths
- [ ] Health / readiness / metrics endpoints respond
- [ ] CI pipeline green on first green-field push

## Non-Goals
- UI / frontend (separate project)
- Migration from an existing API (this is greenfield)
- Multi-region deploy (single-region to start)

## Constraints
- Language: [Python 3.11+ / Node 20+ / Go 1.22+]
- Framework: [FastAPI / Fastify / chi]
- Datastore: [Postgres / SQLite / DynamoDB]
- Deployment target: [container / serverless]

## Tech Stack
- Language: [...]
- Framework: [...]
- Database: [...]
- Observability: [...]
- Container runtime: [...]
