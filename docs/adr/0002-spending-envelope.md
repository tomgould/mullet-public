# 0002. One shared spending envelope plus per-rung limits

Status: Accepted

Context: Several jobs and several paid providers can run in the same day. A cap on one job does not stop their total from growing.

Decision: `budget.envelope.daily` and `.weekly` cap everything paid, across all jobs. `budget.per_rung.<rung>` caps one provider (`kimi`, `openrouter`) over the same windows. Windows are UTC: a day, and a week from Monday 00:00. The window spend is every paid attempt in the ledger that started in the window, running ones included (`mullet/jobs/budget.py`). `ladder.max_spend` still caps one job. The limits are checked before each attempt, which gets an allowance equal to the least of the job's remaining cap and the rung's remaining room. They are checked again inside the attempt: the key proxy answers HTTP 402 once the allowance is spent, and the attempt moves up the ladder.

Consequences: Nothing is limited until a value is set, so `mullet doctor` warns when no envelope is set. A rung with no room is skipped and the reason is recorded. A per-rung limit pauses the job only when no paid rung is left; `mullet job resume <job> --max-spend <dollars>` raises the job's cap. `budget.per_task`, `budget.per_session` and `budget.daily` belong to the older single-task path and gated jobs do not read them.
