---
name: mullet-job
description: Run a big coding job (refactor a repo, rebuild a feature) unattended in a sandboxed clone, with every step checked by Mullet's gates, and hand back a branch to review
triggers:
  - mullet job
  - unattended job
  - refactor the whole repo
  - run this overnight
  - autonomous job
tags:
  - jobs
  - automation
---

# Mullet jobs

Use `mullet job` when someone wants a large piece of work done unattended
on a repository: a refactor across many files, a rebuild of a feature, a
long series of mechanical changes. Mullet runs each step with the user's own
Claude Code in a sandboxed private clone, checks every step with gates it
runs itself (scope, secrets, test weakening, build, tests against the
baseline), and hands back a branch. Nothing touches the user's own branch
until they merge it.

For a small change in the current session, just make it; a job is for work
that should run without anyone watching.

## Before planning

1. Make sure the repo's committed state is what the job should start from.
   Uncommitted changes are not part of a job.
2. Check the repo profile. `mullet job plan` saves a detected one the first
   time and prints its path. Without a test command every node ends up
   "unverified", never "passed", so help the user add one if it's missing.
3. `mullet job plan "<goal>"` asks a read-only Claude Code session for a task
   tree and shows it with the planner's notes. Parents group children; a
   parent passes only when its children have and the full test suite passes
   on their combined change. To change the plan, export it with
   `mullet job show <id> --yaml > plan.yaml`, edit it, and plan again with
   `mullet job plan "<goal>" --plan-file plan.yaml`. `--single` skips the
   planner for a small goal.

## The flow

```bash
mullet job plan "<goal>" [--plan-file plan.yaml | --single] [--no-review]
                 [--executor opencode --executor-model <provider/model>]
                 [--ladder | --no-ladder] [--max-spend 2]
mullet job show <id>          # the plan, budget, worker and the checks that will run
mullet job approve <id>       # only the user approves
mullet job run <id>           # runs in the background
mullet job status <id>        # or: mullet job tail <id> -f
mullet job report <id>        # evidence for every node, from the ledger
mullet job merge <id>         # copies the branch into the repo; the user merges it
```

`stop` pauses a job without losing work; `resume` carries on from the last
checked step. A usage limit pauses the job until the limit resets.

Every attempt gets a brief from Mullet: the goals, the planner's notes, what
earlier steps changed, the checks, a symbol outline and a repo map. After a
gate failure the worker carries on its own session to fix the problem; the
last attempt starts fresh. Planning and review always use Claude. The worker
can be opencode with another model (`--executor opencode`), but only on a
repo whose stored profile lists `opencode` in `allowed_executors`, because
the code then goes to that model's provider. Never add it for the user.

## The delegation ladder

When the user's `~/.config/mullet/config.yaml` has a `ladder:` section and
the stored profile allows `opencode`, `mullet job plan` uses the ladder by
default: each step gets one attempt per rung (free models on verified-public
repos, a local LM Studio model when the machine has room, Kimi, OpenRouter,
Claude), in an order set by Claude's weekly budget: with room, Claude comes
before the paid rungs and leads on design and analysis; when tight, Claude is
the last resort. Planning and review always use Claude. A step that deletes,
kills or sends data is always reviewed by Claude. A provider error moves a
step up without using an attempt, a model that failed a step isn't tried
again elsewhere, and the paid rungs stop at the job's `--max-spend` (default
$2). Over the cap with Claude's week tight, the job pauses and says so;
`mullet job resume <id> --max-spend <dollars>` raises the cap, but only when
the user says so.

```bash
mullet job ladder                         # rungs, which work now, Claude's budget, this repo's public check
mullet ladder report                      # cross-job evidence from the ledger, no model calls
mullet job plan "<goal>" --no-ladder      # one worker for the whole job instead
mullet job retry <id> <step> --rung claude  # a blocked step gets fresh attempts on that rung
mullet job resume <id>                    # then carry on
mullet night on [--until HH:MM]           # lean on the local GPU until 07:30 or the given time
mullet night off                          # back to the daytime ladder
mullet night status                       # on or off, until when, and why it last went off
```

Night mode switches itself off at its end time or when the user is back.
Only the user switches it on.

The report shows each attempt's rung, model, cost and tokens, and a line
saying how many steps passed below Claude and what was spent. Pass those
figures on as they are. Never change `allowed_executors`, `data_class` or
the ladder config for the user, and never retry a blocked step on a cheaper
rung without asking: a blocked step usually needs Claude or a person.

## `mullet_opencode` from Claude Code

A `mullet_opencode` call is a single-step ladder job. The call itself approves
it, so it runs straight away in the foreground with a timeout (default 20
minutes, maximum 60). The step is mechanical, so the ladder starts it on a cheap
rung and only turns review on when `mullet/jobs/ladder.py` flags the step as
risky. The user's stored profile supplies the gates.

When the step PASSED, Mullet computes the job branch's diff against its base and
applies it to the caller's working tree with `git apply --3way`. If any target
file has uncommitted changes, or the base is no longer the caller's HEAD's
ancestor-equal content for those files, the diff is not applied and the result
status is `passed_not_applied`. Nothing is committed or pushed in the caller's
repo.

Possible statuses: `applied` (passed and applied), `passed_not_applied` (passed
but not applied, with a reason), `failed`, `blocked`, `unverified`, `timeout`.

A `providers` hint naming `kimi`, `openrouter`, `free`, `local` or `claude`
limits the ladder to those rungs. The old `model`, `apply` and related arguments
are still accepted but ignored, with a note in the result.

## Rules

- Never approve a plan or merge a branch on the user's behalf. Show them the
  plan (`mullet job show`) and the report, and let them decide.
- Report what the ledger says. A node is passed only when its gates passed;
  "unverified" means some checks could not run, and must be said plainly.
  The executor's own summary is a claim, not evidence.
- If `mullet job run` refuses (no sandbox, no `socat`, no `claude`, no
  `opencode`, an unknown model, an executor the profile doesn't allow), pass
  the reason on as it is. Don't look for a way around the sandbox.
