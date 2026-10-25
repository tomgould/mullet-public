# Python API

`mullet.api` lets a script or CI drive gated jobs without shelling out to `mullet job`. It is a thin layer over the CLI's own code: the same checks, the session strategy and the budget envelope apply, and the messages are the CLI's.

```python
from mullet import api

job = api.plan("add a subtract function with a test", repo=".", plan_file="plan.yaml")
api.approve(job)
result = api.run(job)              # in this process, until done, blocked or paused
print(api.report(job))
if result.status == "done" and not result.nodes_unverified:
    api.merge(job)                 # copies the branch into your repo, nothing else changes
```

Without `plan_file`, a read-only Claude Code session writes the task tree, as `mullet job plan` does. A `job` argument is a job id string or the `JobHandle` that `plan` returns.

## Functions

- `plan(goal, repo=None, *, ref=None, max_spend=None, ladder=None, executor=None, plan_file=None) -> JobHandle`: create a job and leave it awaiting approval.
- `approve(job)`: approve the plan.
- `run(job, *, wait=True, detach=False) -> JobStatusInfo`: run in this process until done, blocked or paused; with `detach=True`, start the background runner and return at once.
- `status(job) -> JobStatusInfo`: the job's state now.
- `report(job) -> str`: the Markdown report.
- `stop(job)`: ask the runner to stop; the job is paused, not lost.
- `resume(job, *, max_spend=None) -> JobStatusInfo`: restart a paused or blocked job in the background, optionally with a new spending cap.
- `merge(job, *, allow_unverified=False, branch=None) -> MergeResult`: copy the finished branch into the repo.
- `list_jobs(limit=20) -> list[JobSummary]`: recent jobs, newest first.

Results are frozen dataclasses of plain values: `id`, `status`, `goal`, `branch`, `nodes_total`, `nodes_passed`, `nodes_unverified`, `nodes_blocked`, `spent_usd` and `error` (`JobStatusInfo` adds `runner_alive`).

## Errors

Every error is a `mullet.errors.MulletError`; those about jobs are `JobError`, and `mullet.api` re-exports them.

| Error | Raised when |
|:------|:------------|
| `JobNotFound` | no job has the id |
| `NotApproved` | the job is still awaiting approval |
| `NotRunnable` | the job's state does not allow it (merge before done, stop with no runner) |
| `RunnerBusy` | a runner already has the job |
| `Unverified` | `merge` met unverified nodes and `allow_unverified` is False |
| `SpendLimit` | a cap is negative, not a number, or does not apply to the job |
| `StrategyOff` | the session strategy turns delegation off |
| `PlanFailed` | the planner could not produce a plan |
| `Unavailable` | the sandbox, Claude Code or opencode is missing, or the profile forbids the worker |

## What is stable

Only the names in `mullet.api.__all__`. Everything else in Mullet, including `mullet.cli` and `mullet.jobs`, may change.
