# Architecture decision records

An ADR records one design decision: what was chosen, why, and what follows from it. Each is short (25 lines at most) and names the code and settings it concerns.

Files are named `NNNN-short-title.md`, numbered from 0001 in the order they are written. A number is never reused. To change a decision, write a new record and set the old one's Status to `Superseded by NNNN`.

Template:

```
# NNNN. Title

Status: Accepted | Superseded by NNNN
Context: the problem and the forces on it.
Decision: what Mullet does, with the real names and defaults.
Consequences: what this makes easy, what it costs, what to watch.
```

| No. | Decision |
|:----|:---------|
| [0001](0001-cost-from-provider-responses.md) | Cost comes from the provider's own responses |
| [0002](0002-spending-envelope.md) | One shared spending envelope plus per-rung limits |
| [0003](0003-toolset-policy.md) | Toolsets for delegated models are a per-provider setting |
| [0004](0004-server-token-and-origin-checks.md) | `mullet serve` needs a token and refuses foreign Host and Origin |
| [0005](0005-protected-paths.md) | Editor, dev container, shell hook and package-manager files are protected |
| [0006](0006-pause-instead-of-judging.md) | A crashed gate or missing verdict pauses the job |
| [0007](0007-sandbox-backends.md) | The sandbox sits behind a backend interface; bubblewrap is the default |
| [0008](0008-planner-dependencies-and-risk.md) | The planner states dependencies, one-way doors and open questions |
| [0009](0009-reviewer-findings-and-standards.md) | The reviewer gives findings and reads the repo's own review standards |
| [0010](0010-worker-idle-timeout-and-grace.md) | Silent workers are stopped and finished ones are given a short grace window |

Settings are in [../CONFIGURATION.md](../CONFIGURATION.md).
