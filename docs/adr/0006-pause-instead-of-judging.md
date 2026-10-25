# 0006. A crashed gate or a missing review verdict pauses the job

Status: Accepted

Context: A bug or outage in Mullet's own checks says nothing about the worker's change. Treating a crash as a pass lets bad work through; treating it as a fail wastes attempts and can bar a good model.

Decision: A gate that raises, or a required review (G7) that gives no verdict, raises `GatePaused` (`mullet/jobs/gates.py`). It is not a verdict. The runner (`mullet/jobs/runner.py`) runs the checks again on the same work up to `MAX_EXECUTOR_CRASHES` times in a row. If they still cannot run, the job is set to paused with the cause, nothing is committed, no attempt is used and the worker's model is not barred. The owner is notified. `mullet job resume` checks the same work again instead of asking the worker for it anew. Work the gates refuse to run against (`UnsafeWorkTree`, an unsafe `copy_in` path) is the change's fault and fails the gate.

Consequences: A step's status still comes only from gate results. A broken environment stops the job visibly instead of producing a wrong verdict. The person fixes the cause and resumes.
