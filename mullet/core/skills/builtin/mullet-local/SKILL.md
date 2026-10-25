---
name: mullet-local
description: Keep a local LLM server such as LM Studio usable and restart it when it hangs or stops answering
triggers:
  - local llm
  - lm studio
  - local model
  - local watchdog
tags:
  - local
  - llm
  - watchdog
---

# Mullet local watchdog

Use `mullet local` when a local model call hangs, LM Studio stops answering,
or the local LLM server needs to be kept healthy without manual intervention.

```bash
mullet local status      # server URL, reachability, latency, models, GPU memory
mullet local check       # decide whether the server needs a restart
mullet local check --fix # restart automatically if the watchdog says so
mullet local restart     # restart the local server
mullet local restart --hard  # stop and start the lms daemon as well
mullet local watchdog --install   # install a systemd user timer that checks every 2 minutes
mullet local watchdog --uninstall # remove the timer
```

If a local model call hangs or LM Studio stops answering, run
`mullet local check --fix`. If that fails to restore the server, run
`mullet local restart --hard`. Use `mullet local status` to see the current
state. Never kill LM Studio processes by hand.
