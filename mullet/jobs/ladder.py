"""The delegation ladder: which worker attempts each step of a job, and in what order.

A job planned with the ladder doesn't send every step to Claude. Each step
gets a list of rungs, cheapest suitable first, and the runner gives each rung
one attempt before moving up. The gates judge every attempt the same way,
whoever made it.

Rungs: ``free`` (an OpenRouter ``:free`` model), ``local`` (a model in LM
Studio), ``kimi`` (Moonshot's API), ``openrouter`` (one rung per configured
model: a cheap paid coder, then OpenRouter's Kimi as a fallback) and
``claude`` (the user's own Claude Code). Every rung but Claude runs through
the opencode executor.

The policy (``plan_rungs`` and ``next_rung``):

- the planner, the reviewer (G7) and every step of a repo whose profile
  says ``ladder: claude-only`` are Claude only;
- each step is mechanical, checkable (a design step with acceptance
  commands, a narrow scope and nothing risky) or claude-first (analysis, any
  other design step, a one-way step, a parent's integration fix). The order of rungs comes
  from Claude's budget level (``ORDERS``):

  - low (Claude's week has room): mechanical ``free, local, claude, kimi,
    openrouter`` (the paid rungs after Claude, which is paid for already);
    everything else ``claude, kimi``;
  - medium: mechanical ``free, local, kimi, openrouter, claude``; checkable
    ``kimi, openrouter, claude``; claude-first ``claude, kimi``;
  - high: ``free, local, kimi, openrouter, claude`` for mechanical and
    checkable steps, ``kimi, openrouter, claude`` for the rest, with Claude
    the last resort (one try);
  - no fresh figures: the last known level if it is under 6 hours old
    (kept in the state dir), else medium. The runner records Claude's rate
    limits from every ``claude -p`` stream it sees, so jobs keep them fresh;
  - night mode (``mullet.jobs.night``): checkable steps at medium try local
    first, cheap attempts get longer and more of them (``apply_night``), and
    the local rung waits for headroom (``machine_state(wait_s=...)``);

- ``free`` needs the public check (``check_public``, cached per repo and base
  for an hour), ``local`` needs LM Studio and machine-guard room
  (``machine_state``), the paid rungs (``kimi``, ``openrouter``) stop at the
  job's ``max_spend``, and a rung whose executor the profile doesn't allow,
  or whose provider has no key, is skipped. ``ladder.start`` sets the
  cheapest rung each kind may use;
- one attempt per rung (Claude gets the rest of the job's attempts, or one
  when its week is tight); a model that failed a step's gates is never tried
  again on another provider; a step gets at most ``max_below_claude``
  counted attempts below Claude, before or after Claude's own, and then only
  Claude; risky steps (``is_risky``: one-way, or deletes, kills or sends
  data) always get G7. Every skip is recorded with its reason;
- every rung whose requests go to OpenRouter gets ``data_collection: deny``
  and the quantisation filter (``request_fields``), by where it sends, not by
  its name; the free rung on its ``:free`` model is the one exception, since
  it only ever sees code that is public already.

The config is the ``ladder:`` key of the user's own Mullet config
(``config_file("config.yaml")``); never a file in the repo or the working
directory. See ``load_config`` for the keys.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from mullet.jobs import night
from mullet.jobs.night import NightSettings, NightState
from mullet.jobs.profile import Profile
from mullet.jobs.types import NodeSpec

logger = logging.getLogger(__name__)

RUNG_ORDER: tuple[str, ...] = ("free", "local", "kimi", "openrouter", "claude")
PAID_RUNGS = frozenset({"kimi", "openrouter"})
# opencode providers that run on this machine, so their models cost nothing.
LOCAL_PROVIDERS = frozenset({"lmstudio", "ollama"})

DEFAULT_KIMI_MODEL = "moonshotai/kimi-k2.7-code"
DEFAULT_OPENROUTER_MODELS: tuple[str, ...] = ("openrouter/qwen/qwen3-coder", "openrouter/moonshotai/kimi-k2.7-code")
DEFAULT_LOCAL_URL = "http://127.0.0.1:1234"
DEFAULT_CLAUDE_LOW = 0.5
DEFAULT_CLAUDE_HIGH = 0.8
DEFAULT_MAX_SPEND = 2.0
# Dollars per million tokens, for a paid attempt whose provider reported no
# cost and whose model has no configured price. Deliberately on the high side,
# so an unpriced model reaches the spending cap early rather than late.
DEFAULT_FALLBACK_PRICE: dict[str, float] = {"input": 2.0, "output": 8.0}
# Where each kind of step starts. Real plans mark most steps "design", so a
# design step may start on Kimi, but only when its checks can judge it (it
# has acceptance commands and a narrow scope) and it isn't risky.
# The cheapest rung each kind of step may use; the budget level's order
# (ORDERS) applies on top. Analysis steps never go to free or local models.
DEFAULT_START: dict[str, str] = {"mechanical": "free", "design": "free", "analysis": "kimi"}
DEFAULT_NARROW_GLOBS = 4
# OpenRouter host routing for every rung whose requests go to OpenRouter,
# whatever the rung is called, except the free rung (see
# ``LadderConfig.request_fields``). Hosts that keep or train on prompts are
# always left out, since private code goes there. By default so are hosts
# serving a 4-bit, 6-bit or int8 copy, which code noticeably worse, and hosts
# that don't say which copy they serve ("unknown"). ``routing:`` in the config
# sets other fields per model; a ``quantizations`` list there replaces this one,
# so a user who accepts unknown hosts lists "unknown" in it.
DEFAULT_OPENROUTER_ROUTING: dict[str, Any] = {
    "data_collection": "deny",
    "quantizations": ["fp8", "fp16", "bf16", "fp32"],
}
OPENROUTER_PROVIDER = "openrouter"  # the opencode provider id of models served through OpenRouter
_ROUTING_LISTS = ("order", "only", "ignore", "quantizations")
DEFAULT_MAX_BELOW_CLAUDE = 3  # attempts below Claude before Claude gets its turn
DEFAULT_CHEAP_MINUTES = 10.0  # wall clock per free or local attempt
LEVEL_MAX_AGE_S = 6 * 3600  # how long a known budget level stands in when there are no figures
PUBLIC_CACHE_S = 3600.0  # how long a public check holds, per repo and base commit
# Words in a step's goal, acceptance or scope that mark code which deletes,
# kills or sends data. Such a step always gets the Claude reviewer (G7), and
# a design step with one isn't "checkable".
DEFAULT_RISKY_WORDS: tuple[str, ...] = (
    "delete", "remove", "rm", "rmtree", "unlink", "kill", "terminate", "signal", "wipe", "truncate", "drop",
    "push", "upload", "send",
)
# Groups of steps, by how much they need Claude.
MECHANICAL, CHECKABLE, CLAUDE_FIRST = "mechanical", "checkable", "claude-first"
# The order of rungs per budget level and group. Claude is already paid for,
# so when its week has room it comes before the paid rungs.
ORDERS: dict[str, dict[str, tuple[str, ...]]] = {
    "low": {MECHANICAL: ("free", "local", "claude", "kimi", "openrouter"),
            CHECKABLE: ("claude", "kimi"), CLAUDE_FIRST: ("claude", "kimi")},
    "medium": {MECHANICAL: ("free", "local", "kimi", "openrouter", "claude"),
               CHECKABLE: ("kimi", "openrouter", "claude"), CLAUDE_FIRST: ("claude", "kimi")},
    "high": {MECHANICAL: ("free", "local", "kimi", "openrouter", "claude"),
             CHECKABLE: ("free", "local", "kimi", "openrouter", "claude"),
             CLAUDE_FIRST: ("kimi", "openrouter", "claude")},
}

GITHUB_API = "https://api.github.com"
GITHUB_TIMEOUT_S = 10.0
LMSTUDIO_TIMEOUT_S = 3.0

_MODEL_ID = re.compile(r"^[A-Za-z0-9][\w.~@-]*/\S+$")
_SHA = re.compile(r"^[0-9a-f]{7,64}$")
_GITHUB_REMOTE = re.compile(
    r"^(?:https://github\.com/|ssh://git@github\.com/|git@github\.com:)"
    r"([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$")


# --- Rungs and config ---------------------------------------------------------------


@dataclass(frozen=True)
class Rung:
    """One worker on the ladder: a rung name and, for opencode rungs, the model and where it is."""

    name: str
    model: str | None = None  # an opencode provider/model id; None for Claude (the job's worker model)
    upstream: str | None = None  # the provider's base URL when it isn't the configured one (LM Studio)

    @property
    def executor(self) -> str:
        """The executor name the profile's ``allowed_executors`` must list."""
        return "claude-code" if self.name == "claude" else "opencode"

    @property
    def paid(self) -> bool:
        """True for the rungs the spending cap applies to."""
        return self.name in PAID_RUNGS

    @property
    def key(self) -> str:
        """Identifies the rung within a job: its name and model."""
        return f"{self.name}:{self.model or ''}"

    @property
    def slug(self) -> str:
        """The rung as a file name, for its executor's logs and home."""
        return re.sub(r"[^A-Za-z0-9._-]+", "-", self.key).strip("-")[:80]

    def label(self) -> str:
        """Like ``kimi (moonshotai/kimi-k2.7-code)``, or ``claude``."""
        return f"{self.name} ({self.model})" if self.model else self.name


def model_rung(model: str) -> Rung:
    """The rung a single opencode worker's model counts as, for spending and the free-model check.

    A ``:free`` model is the free rung and an LM Studio or ollama model the local
    rung, neither paid; an OpenRouter model is the openrouter rung, and any other
    provider's model the kimi rung (a provider billed directly).
    """
    if model.endswith(":free"):
        return Rung("free", model)
    provider = model.split("/", 1)[0]
    if provider in LOCAL_PROVIDERS:
        return Rung("local", model)
    return Rung("openrouter" if provider == "openrouter" else "kimi", model)


@dataclass
class LadderConfig:
    """The ``ladder:`` section of the user's config, with defaults filled in."""

    configured: bool = False  # the user's config has a ladder: key
    kimi_model: str | None = DEFAULT_KIMI_MODEL
    openrouter_models: list[str] = field(default_factory=lambda: list(DEFAULT_OPENROUTER_MODELS))
    free_model: str | None = None
    local_model: str | None = None
    local_url: str = DEFAULT_LOCAL_URL
    claude_low: float = DEFAULT_CLAUDE_LOW
    claude_high: float = DEFAULT_CLAUDE_HIGH
    max_spend: float = DEFAULT_MAX_SPEND
    prices: dict[str, dict[str, float]] = field(default_factory=dict)
    fallback_price: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FALLBACK_PRICE))
    start: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_START))  # kind -> first rung
    narrow_globs: int = DEFAULT_NARROW_GLOBS  # a checkable design step has at most this many
    risky_words: tuple[str, ...] = DEFAULT_RISKY_WORDS
    max_below_claude: int = DEFAULT_MAX_BELOW_CLAUDE
    cheap_minutes: float = DEFAULT_CHEAP_MINUTES
    local_by_day: bool = True  # False: the local rung only runs while night mode is on
    routing: dict[str, dict[str, Any]] = field(default_factory=dict)  # model -> OpenRouter provider fields
    source: str | None = None
    problems: list[str] = field(default_factory=list)

    def rungs(self) -> list[Rung]:
        """Every configured rung, in ladder order."""
        out: list[Rung] = []
        if self.free_model:
            out.append(Rung("free", self.free_model))
        if self.local_model:
            out.append(Rung("local", self.local_model, upstream=self.local_url.rstrip("/") + "/v1"))
        if self.kimi_model:
            out.append(Rung("kimi", self.kimi_model))
        out += [Rung("openrouter", m) for m in self.openrouter_models]
        out.append(Rung("claude"))
        return out

    def request_fields(self, rung: Rung) -> dict[str, Any] | None:
        """Fields the key proxy sets on the rung's requests: OpenRouter host routing, by where the requests go.

        Every rung whose model is served through OpenRouter gets it, whatever
        the rung is called (a ``kimi:`` set to an ``openrouter/...`` model
        included). The one exception is the free rung on its ``:free`` model:
        it only runs on code the public check found public already, which is
        what makes a free host that may keep prompts acceptable, so it carries
        no host filter that could leave a free model with no host at all. A
        ``:free`` model on any other rung is filtered like a paid one.
        """
        if not rung.model or rung.upstream is not None or not is_openrouter_model(rung.model):
            return None
        if rung.name == "free" and rung.model.endswith(":free"):
            return None
        provider = {**DEFAULT_OPENROUTER_ROUTING, **self.routing.get(rung.model, {})}
        provider["data_collection"] = "deny"
        # `usage.include` makes OpenRouter state each request's real cost in the response.
        return {"provider": provider, "usage": {"include": True}}

    def price(self, model: str | None) -> dict[str, float]:
        """Dollars per million tokens for `model`: configured, else the fallback."""
        return self.prices.get(model or "") or self.fallback_price


def is_openrouter_model(model: str | None) -> bool:
    """True when an opencode model id is served through OpenRouter (``openrouter/<vendor>/<model>``)."""
    return bool(model) and str(model).split("/", 1)[0] == OPENROUTER_PROVIDER


def config_path() -> Path:
    """The user's Mullet config file, which holds the ``ladder:`` key."""
    from mullet.core.paths import config_file

    return config_file("config.yaml")


def _model(value: Any, where: str, problems: list[str]) -> str | None:
    if value is None or value is False or (isinstance(value, str) and value.strip().lower() in ("", "off", "none")):
        return None
    if isinstance(value, dict):
        value = value.get("model")
        if value is None:
            return None
    if isinstance(value, str) and _MODEL_ID.match(value.strip()):
        return value.strip()
    problems.append(f"ladder.{where}: {value!r} is not an opencode model id like provider/model; ignored")
    return None


def _number(value: Any, where: str, default: float, problems: list[str], *, upper: float | None = None) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if not math.isfinite(number) or number < 0 or (upper is not None and number > upper):
        problems.append(f"ladder.{where}: {value!r} is not a valid number; using {default:g}")
        return default
    return number


def _routing(value: Any, problems: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if value is None:
        return out
    if not isinstance(value, dict):
        problems.append("ladder.routing: expected a mapping of OpenRouter model id to host settings; ignored")
        return out
    for model, entry in value.items():
        if not isinstance(entry, dict):
            problems.append(f"ladder.routing.{model}: expected a mapping; ignored")
            continue
        fields: dict[str, Any] = {}
        for key, item in entry.items():
            if key in _ROUTING_LISTS and isinstance(item, list) and all(isinstance(i, str) for i in item):
                fields[key] = list(item)
            elif key == "allow_fallbacks" and isinstance(item, bool):
                fields[key] = item
            elif key == "sort" and item in ("price", "throughput", "latency"):
                fields[key] = item
            else:
                problems.append(f"ladder.routing.{model}.{key}: not a host setting Mullet passes on; ignored")
        out[str(model)] = fields
    return out


def _prices(value: Any, problems: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if value is None:
        return out
    if not isinstance(value, dict):
        problems.append("ladder.prices: expected a mapping of model id to {input, output}; ignored")
        return out
    for model, price in value.items():
        if not isinstance(price, dict):
            problems.append(f"ladder.prices.{model}: expected {{input, output}}; ignored")
            continue
        entry = {}
        for part in ("input", "output", "cache_read"):
            if part in price:
                entry[part] = _number(price[part], f"prices.{model}.{part}", 0.0, problems)
        out[str(model)] = entry
    return out


def parse_config(data: Any, source: str | None = None) -> LadderConfig:
    """A ``LadderConfig`` from the value of the ``ladder:`` key (None when the key is missing).

    Keys, all optional::

        ladder:
          kimi: moonshotai/kimi-k2.7-code          # or {model: ...}, or off
          openrouter: [openrouter/qwen/qwen3-coder, openrouter/moonshotai/kimi-k2.7-code]
          free: openrouter/qwen/qwen3-coder:free   # must be an OpenRouter :free model
          local: lmstudio/qwen3.6-35b-a3b          # or {model: ..., url: http://127.0.0.1:1234}
          claude_low: 0.5                          # paced weekly pressure below this: Claude before paid rungs
          claude_high: 0.8                         # above this: Claude only as the last resort
          max_spend: 2.0                           # dollars per job on the paid rungs
          prices:                                  # $/M tokens, used when a provider reports no cost
            moonshotai/kimi-k2.7-code: {input: 0.6, output: 2.5}
          start:                                   # the cheapest rung each kind of step may use
            mechanical: cheapest                   # free, local, kimi, openrouter or claude
            design: cheapest
            analysis: kimi
          narrow_globs: 4                          # most scope globs a checkable design step may have
          risky_words: [delete, kill, send, ...]   # mark steps that always get the G7 review
          max_below_claude: 3                      # most counted attempts below Claude per step
          cheap_minutes: 10                        # wall clock per free or local attempt
          local_by_day: true                       # false: the local rung only runs in night mode
          routing:                                 # OpenRouter host choice per model (data_collection is always deny)
            openrouter/z-ai/glm-5.3: {only: [z-ai]}  # also order, ignore, quantizations, allow_fallbacks, sort
            openrouter/qwen/qwen3-coder:             # a quantizations list replaces the default
              quantizations: [fp8, fp16, bf16, fp32, unknown]  # the default leaves out unknown, fp4, int4, int8

    A value it can't use is dropped with a note in ``problems``, never guessed.
    """
    config = LadderConfig(configured=data is not None, source=source)
    if data is None:
        return config
    if data is True or (isinstance(data, str) and data.strip().lower() in ("on", "default")):
        return config
    if not isinstance(data, dict):
        config.problems.append("ladder: expected a mapping; using the defaults")
        return config
    problems = config.problems
    if "kimi" in data:
        config.kimi_model = _model(data["kimi"], "kimi", problems)
    if "openrouter" in data:
        raw = data["openrouter"]
        if isinstance(raw, dict):
            raw = raw.get("models")
        if raw is None or raw is False:
            config.openrouter_models = []
        else:
            items = raw if isinstance(raw, list) else [raw]
            models = [_model(m, "openrouter", problems) for m in items]
            config.openrouter_models = list(dict.fromkeys(m for m in models if m))
    free = _model(data.get("free"), "free", problems)
    if free and not (free.startswith("openrouter/") and free.endswith(":free")):
        problems.append(f"ladder.free: {free!r} is not an OpenRouter :free model; the free rung is off")
        free = None
    config.free_model = free
    local_raw = data.get("local")
    local = _model(local_raw, "local", problems)
    if local and not local.startswith("lmstudio/"):
        problems.append(f"ladder.local: {local!r} is not an LM Studio model (lmstudio/<id>); the local rung is off")
        local = None
    config.local_model = local
    if isinstance(local_raw, dict) and local_raw.get("url"):
        url = str(local_raw["url"]).rstrip("/")
        parts = urllib.parse.urlsplit(url)
        if parts.scheme == "http" and parts.hostname in ("127.0.0.1", "localhost", "::1") and parts.path in ("", "/"):
            config.local_url = url
        else:
            problems.append(f"ladder.local.url: {url!r} must be http://127.0.0.1:<port> or localhost; "
                            f"using {DEFAULT_LOCAL_URL}")
    config.claude_low = _number(data.get("claude_low"), "claude_low", DEFAULT_CLAUDE_LOW, problems, upper=1.0)
    config.claude_high = _number(data.get("claude_high"), "claude_high", DEFAULT_CLAUDE_HIGH, problems, upper=1.0)
    if config.claude_high < config.claude_low:
        problems.append("ladder: claude_high is below claude_low; using the defaults for both")
        config.claude_low, config.claude_high = DEFAULT_CLAUDE_LOW, DEFAULT_CLAUDE_HIGH
    config.max_spend = _number(data.get("max_spend"), "max_spend", DEFAULT_MAX_SPEND, problems)
    start = data.get("start")
    if start is not None:
        if not isinstance(start, dict):
            problems.append("ladder.start: expected a mapping of step kind to rung; using the defaults")
        else:
            for kind, rung in start.items():
                rung = "free" if rung == "cheapest" else rung
                if kind not in DEFAULT_START:
                    problems.append(f"ladder.start.{kind}: not a step kind (mechanical, design, analysis); ignored")
                elif rung not in RUNG_ORDER:
                    problems.append(f"ladder.start.{kind}: {rung!r} is not a rung ({', '.join(RUNG_ORDER)}); ignored")
                else:
                    config.start[str(kind)] = str(rung)
    if data.get("max_below_claude") is not None:
        config.max_below_claude = int(_number(data["max_below_claude"], "max_below_claude",
                                              DEFAULT_MAX_BELOW_CLAUDE, problems))
    if "local_by_day" in data:
        if isinstance(data["local_by_day"], bool):
            config.local_by_day = data["local_by_day"]
        else:
            problems.append("ladder.local_by_day: expected true or false; using true")
    config.cheap_minutes = _number(data.get("cheap_minutes"), "cheap_minutes", DEFAULT_CHEAP_MINUTES, problems)
    if config.cheap_minutes <= 0:
        problems.append(f"ladder.cheap_minutes: must be above 0; using {DEFAULT_CHEAP_MINUTES:g}")
        config.cheap_minutes = DEFAULT_CHEAP_MINUTES
    if data.get("narrow_globs") is not None:
        config.narrow_globs = int(_number(data["narrow_globs"], "narrow_globs", DEFAULT_NARROW_GLOBS, problems))
    words = data.get("risky_words")
    if words is not None:
        if isinstance(words, list) and all(isinstance(w, str) for w in words):
            config.risky_words = tuple(w.lower() for w in words if w.strip())
        else:
            problems.append("ladder.risky_words: expected a list of words; using the defaults")
    config.prices = _prices(data.get("prices"), problems)
    config.routing = _routing(data.get("routing"), problems)
    if "fallback_price" in data:
        fallback = _prices({"fallback": data["fallback_price"]}, problems).get("fallback")
        if fallback:
            config.fallback_price = {**DEFAULT_FALLBACK_PRICE, **fallback}
    return config


def apply_night(config: LadderConfig, state: NightState,
                settings: NightSettings = night.SETTINGS) -> LadderConfig:
    """The config for a step choice while night mode is `state`: a copy with longer cheap
    attempts and more of them below Claude when it is on, else `config` itself.

    Prices, the spending cap, Claude's thresholds, the models and ``start`` are
    left alone.
    """
    if not state.on:
        return config
    return replace(config, cheap_minutes=config.cheap_minutes * settings.cheap_minutes_factor,
                   max_below_claude=config.max_below_claude + settings.extra_below_claude,
                   problems=list(config.problems))


def load_config(path: Path | None = None) -> LadderConfig:
    """The ladder config from the user's Mullet config file; defaults when there is none."""
    target = path or config_path()
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return LadderConfig()
    except OSError as exc:
        config = LadderConfig()
        config.problems.append(f"{target}: {exc}")
        return config
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        config = LadderConfig()
        config.problems.append(f"{target}: invalid YAML ({exc}); the ladder uses its defaults")
        return config
    if not isinstance(data, dict):
        return LadderConfig()
    return parse_config(data.get("ladder"), str(target))


# --- Claude's budget -----------------------------------------------------------------


@dataclass(frozen=True)
class BudgetState:
    """How much room Claude's weekly window has: ``low``, ``medium`` or ``high`` pressure."""

    level: str
    weekly_pressure: float | None = None  # paced, 0 to 1; None when there are no recent figures
    stale_s: float | None = None  # set when this is the last known level, not a fresh one

    def describe(self) -> str:
        """One line for people."""
        if self.weekly_pressure is None:
            return "Claude budget: no recent usage figures (treated as medium)"
        known = f", last known {self.stale_s / 3600:.1f}h ago" if self.stale_s is not None else ""
        return f"Claude budget: {self.level} (paced weekly pressure {self.weekly_pressure:.2f}{known})"


def budget_state(config: LadderConfig, limits: Any) -> BudgetState:
    """The budget level from `limits` (``fetch_usage()``'s result, None when there are no figures)."""
    if limits is None:
        return BudgetState("medium", None)
    from mullet.hud.guard import pressure_components

    try:
        _, weekly = pressure_components(limits)
    except Exception as exc:  # odd figures must never stop a job
        logger.warning("ladder: can't read Claude's usage figures: %s", exc)
        return BudgetState("medium", None)
    if weekly < config.claude_low:
        return BudgetState("low", weekly)
    if weekly > config.claude_high:
        return BudgetState("high", weekly)
    return BudgetState("medium", weekly)


def current_usage() -> Any:
    """Claude's rate-limit figures as the statusline last recorded them, or None."""
    from mullet.hud.usage import fetch_usage

    try:
        return fetch_usage()
    except Exception as exc:
        logger.warning("ladder: usage figures unavailable: %s", exc)
        return None


# --- The machine, for the local rung ----------------------------------------------------


@dataclass(frozen=True)
class MachineState:
    """Whether the local rung can run now, and why or why not."""

    local_ok: bool
    reason: str = ""
    model_state: str | None = None  # LM Studio's state for the model: "loaded", "not-loaded", ...


def lmstudio_models(url: str, timeout: float = LMSTUDIO_TIMEOUT_S) -> dict[str, str]:
    """LM Studio's models and their states, from ``GET /api/v0/models``. Never sends a completion.

    Raises OSError or ValueError when LM Studio doesn't answer or answers nonsense.
    """
    with urllib.request.urlopen(url.rstrip("/") + "/api/v0/models", timeout=timeout) as resp:  # noqa: S310
        data = json.loads(resp.read(4 * 1024 * 1024))
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("no model list in LM Studio's answer")
    return {str(m.get("id")): str(m.get("state") or "unknown") for m in items if isinstance(m, dict) and m.get("id")}


def machine_state(config: LadderConfig, *, models: Callable[[str], dict[str, str]] | None = None,
                  holder: Callable[[], Any] | None = None, sample: Callable[[], Any] | None = None,
                  wait_s: float = 0.0, clock: Callable[[], float] | None = None,
                  sleep: Callable[[float], None] | None = None) -> MachineState:
    """Whether the local rung may run: a model is configured, nothing else holds the GPU lease,
    LM Studio answers and has the model, and the machine guard finds headroom.

    VRAM LM Studio already holds counts as free when the model is loaded, since
    the step needs no more. With `wait_s` above 0 (night mode) it waits up to
    that long for headroom (``machine.wait_for_headroom``) instead of taking
    one sample; the limits are the same either way. `models`, `holder`,
    `sample`, `clock` and `sleep` replace ``lmstudio_models``,
    ``machine.current_holder``, ``machine.headroom``, and the wait's clock and
    sleep.
    """
    from mullet.jobs import machine

    if not config.local_model:
        return MachineState(False, "no local model configured (ladder.local)")
    who = (holder or machine.current_holder)()
    if who is not None and getattr(who, "pid", None) != os.getpid():
        return MachineState(False, f"the GPU lease is held: {who.describe()}")
    lm_id = config.local_model.split("/", 1)[1]
    try:
        listed = (models or lmstudio_models)(config.local_url)
    except (OSError, ValueError) as exc:
        return MachineState(False, f"LM Studio doesn't answer at {config.local_url}: {type(exc).__name__}: {exc}")
    state = listed.get(lm_id)
    if state is None:
        return MachineState(False, f"LM Studio has no model {lm_id!r}")
    measure = sample or machine.headroom
    if wait_s > 0:
        # LM Studio's pids are read from each snapshot, so its VRAM counts as
        # free while waiting just as it does for a single sample.
        own: set[int] = set()

        def tracked() -> Any:
            snapshot = measure()
            own.clear()
            if state == "loaded":
                own.update(p.pid for p in snapshot.lm_studio_processes)
            return snapshot

        try:
            room = machine.wait_for_headroom(timeout=wait_s, sample=tracked, own_pids=own, clock=clock, sleep=sleep)
        except machine.NoHeadroom as exc:
            return MachineState(False, "not enough headroom: " + "; ".join(s.explain() for s in exc.shortfalls),
                                state)
        # Without a GPU the wait goes ahead anyway; the local rung still needs every limit met.
        short = room.shortfalls(machine.limits(), own_pids=own)
    else:
        room = measure()
        pids = [p.pid for p in room.lm_studio_processes] if state == "loaded" else []
        short = room.shortfalls(machine.limits(), own_pids=pids)
    if short:
        return MachineState(False, "not enough headroom: " + "; ".join(s.explain() for s in short), state)
    return MachineState(True, f"LM Studio has {lm_id} ({state})", state)


# --- Spending -----------------------------------------------------------------------


@dataclass
class SpendState:
    """What the job has spent on the paid rungs, against its cap."""

    cap: float
    spent: float = 0.0
    by_rung: dict[str, float] = field(default_factory=dict)
    estimated: bool = False  # some of it is tokens times a price, not a reported cost
    # Why `cap` is what it is when a shared limit (budget.envelope) lowered it below the job's own.
    reason: str = ""
    # Dollars left under budget.per_rung for a paid rung, and the reason it runs out; unlisted rungs are unlimited.
    rung_left: dict[str, float] = field(default_factory=dict)
    rung_reasons: dict[str, str] = field(default_factory=dict)

    @property
    def capped(self) -> bool:
        """True once the paid rungs are used up for this job."""
        return self.spent >= self.cap


def attempt_cost(usage: dict[str, Any] | None, config: LadderConfig) -> tuple[float, bool]:
    """(dollars, estimated) for one attempt: the reported cost, else tokens times the price.

    Only paid rungs cost anything here; Claude on a subscription and the free
    and local rungs are 0.
    """
    usage = usage or {}
    if usage.get("rung") not in PAID_RUNGS:
        return 0.0, False
    if usage.get("cost_reported"):
        try:
            reported = float(usage.get("total_cost_usd") or 0.0)
        except (TypeError, ValueError):
            reported = 0.0
        if reported > 0:
            return reported, False
        # A paid model reporting $0 for real tokens has no price in the
        # catalogue: count it from the tokens, or the cap would never fire.
    cost = token_cost(usage.get("tokens"), config.price(usage.get("model")))
    return cost, cost > 0


def token_cost(tokens: Any, price: dict[str, float]) -> float:
    """Dollars for opencode's token counts at `price` ($ per million tokens)."""
    if not isinstance(tokens, dict) or not tokens:
        return 0.0

    def count(*names: str) -> float:
        return float(sum(int(tokens.get(n) or 0) for n in names if isinstance(tokens.get(n), (int, float))))

    return (count("input", "cache_write") * price.get("input", 0.0)
            + count("output", "reasoning") * price.get("output", 0.0)
            + count("cache_read") * price.get("cache_read", price.get("input", 0.0))) / 1_000_000


def spend_state(usages: list[dict[str, Any] | None], cap: float, config: LadderConfig) -> SpendState:
    """The job's spend from its attempts' usage records."""
    state = SpendState(cap=cap)
    for usage in usages:
        cost, estimated = attempt_cost(usage, config)
        if cost:
            rung = str((usage or {}).get("rung"))
            state.spent += cost
            state.by_rung[rung] = state.by_rung.get(rung, 0.0) + cost
            state.estimated = state.estimated or estimated
    return state


# --- The public check, for the free rung ----------------------------------------------------


@dataclass(frozen=True)
class PublicCheck:
    """Whether a repo's code may go to a free model: only when it is public already.

    `transient` marks a failure that says nothing about the repo (GitHub
    unreachable, a 5xx): it skips the free rung for one attempt and is never
    cached.
    """

    public: bool
    reason: str
    slug: str | None = None  # owner/repo on GitHub
    default_branch: str | None = None
    checked_at: float = 0.0
    transient: bool = False

    def to_dict(self) -> dict[str, Any]:
        """For the job's event log and the cache."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PublicCheck:
        """Back from the event log or the cache."""
        return cls(bool(data.get("public")), str(data.get("reason") or ""), data.get("slug"),
                   data.get("default_branch"), float(data.get("checked_at") or 0.0), bool(data.get("transient")))


class GitHubError(RuntimeError):
    """GitHub's API didn't give a usable answer. `transient` unless GitHub said the repo isn't there."""

    def __init__(self, message: str, transient: bool = True):
        super().__init__(message)
        self.transient = transient


def github_json(path: str, timeout: float = GITHUB_TIMEOUT_S) -> dict[str, Any]:
    """GET ``https://api.github.com<path>`` without credentials. Raises GitHubError on any failure."""
    req = urllib.request.Request(GITHUB_API + path, headers={
        "Accept": "application/vnd.github+json", "User-Agent": "mullet-ladder", "X-GitHub-Api-Version": "2022-11-28"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = json.loads(resp.read(2 * 1024 * 1024))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise GitHubError("not found (private, or not on GitHub)", transient=False) from exc
        raise GitHubError(f"HTTP {exc.code}") from exc
    except (OSError, ValueError) as exc:
        raise GitHubError(f"unreachable: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise GitHubError("unexpected answer")
    return data


def github_slug(remote_url: str) -> str | None:
    """``owner/repo`` for a GitHub remote URL, or None for anything else."""
    match = _GITHUB_REMOTE.match(remote_url.strip())
    if not match or any(set(part) == {"."} for part in match.groups()):
        return None  # "." and ".." would climb the API path
    return f"{match.group(1)}/{match.group(2)}"


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null", *args],
                          capture_output=True, text=True, timeout=60, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {args[0]} failed")
    return proc.stdout


def check_public(repo: Path, base_sha: str, profile: Profile, *,
                 fetch: Callable[[str], dict[str, Any]] = github_json,
                 git: Callable[..., str] = _git, now: Callable[[], float] = time.time) -> PublicCheck:
    """Whether the job's code is public already, so a free model may see it. Fails closed.

    All of: the stored profile says ``data_class: public`` (a repo's own file
    can't); no ``copy_in`` files; no ``.gitmodules`` or submodules at the base
    commit; ``origin`` is on GitHub, and GitHub says the repo is neither private
    nor a fork; and the base commit is on ``origin``'s default branch there.
    """
    def no(reason: str, slug: str | None = None, transient: bool = False) -> PublicCheck:
        return PublicCheck(False, reason, slug, None, now(), transient)

    if profile.data_class != "public":
        return no("the repo's data class is private (only your stored profile can make it public)")
    if profile.copy_in:
        return no("the profile copies local files into the clone (copy_in)")
    if not _SHA.match(base_sha or ""):
        return no("the base is not a commit hash")
    try:
        if git(repo, "ls-tree", base_sha, "--", ".gitmodules").strip():
            return no("the base commit has a .gitmodules file")
        if any(line.startswith("160000 ") for line in git(repo, "ls-tree", "-r", base_sha).splitlines()):
            return no("the base commit has submodules")
        remote = git(repo, "remote", "get-url", "origin").strip()
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        return no(f"git: {exc}")
    slug = github_slug(remote)
    if slug is None:
        return no("origin is not a GitHub repository")
    try:
        info = fetch(f"/repos/{slug}")
    except GitHubError as exc:
        return no(f"GitHub's API: {exc}", slug, exc.transient)
    if info.get("private") is not False or info.get("visibility", "public") != "public":
        return no("GitHub says the repo is private", slug)
    if info.get("fork") is not False:
        return no("the repo is a fork, so its code may not be its owner's to share", slug)
    default = info.get("default_branch")
    if not isinstance(default, str) or not default:
        return no("GitHub didn't say which branch is the default", slug, True)
    try:
        compared = fetch(f"/repos/{slug}/compare/{urllib.parse.quote(default, safe='')}...{base_sha}")
    except GitHubError as exc:
        # A 404 here means GitHub doesn't have the commit: unpushed, so not public.
        return no(f"can't confirm the base commit is on {default}: {exc}", slug, exc.transient)
    if compared.get("status") not in ("behind", "identical"):
        return no(f"the base commit isn't on origin/{default} (unpushed work stays private)", slug)
    return PublicCheck(True, f"public on GitHub ({slug}), base commit on {default}", slug, default, now())


def _public_cache_path(repo: Path, base_sha: str) -> Path:
    from mullet.core.paths import state_dir

    digest = hashlib.sha256(f"{Path(repo).resolve()}\0{base_sha}".encode()).hexdigest()[:24]
    return state_dir() / "ladder" / "public" / f"{digest}.json"


def cached_public_check(repo: Path, base_sha: str, profile: Profile, *,
                        check: Callable[[Path, str, Profile], PublicCheck] = check_public,
                        now: Callable[[], float] = time.time, max_age: float = PUBLIC_CACHE_S) -> PublicCheck:
    """The public check, from the state-dir cache when it is under `max_age` old (per repo and base).

    A transient failure is returned but not cached, so the next free attempt
    asks again. The profile's own conditions (data class, copy_in) are checked
    every time, since they don't come from GitHub.
    """
    if profile.data_class != "public" or profile.copy_in:
        return check(repo, base_sha, profile)
    path = _public_cache_path(repo, base_sha)
    try:
        cached = PublicCheck.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if not cached.transient and 0 <= now() - cached.checked_at < max_age:
            return cached
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    found = check(repo, base_sha, profile)
    if not found.transient:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(found.to_dict()), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("ladder: can't cache the public check: %s", exc)
    return found


# --- Keys ---------------------------------------------------------------------------


def rung_has_key(rung: Rung) -> bool:
    """True when the rung's provider has an API key Mullet can use (Claude and local need none)."""
    if rung.name in ("claude", "local") or not rung.model:
        return True
    from mullet.jobs.executors import opencode as oc

    provider = oc.provider_of(rung.model)
    xdg = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    config_dir = Path(xdg) / "opencode"
    catalogue = oc.user_catalogue()
    names = oc.key_names(provider, config_dir, catalogue if catalogue.is_file() else None)
    return bool(oc.resolve_keys(names)) or bool(oc.configured_key(provider, config_dir))


# --- The policy ---------------------------------------------------------------------


@dataclass
class RungChoice:
    """The rungs a step may use, in order, and every rung left out with the reason."""

    rungs: list[Rung]
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (rung label, reason)
    group: str = CLAUDE_FIRST
    capped: bool = False  # paid rungs were left out because the job's spending cap is reached

    def to_dict(self) -> dict[str, Any]:
        """For the job's event log."""
        return {"rungs": [r.key for r in self.rungs], "skipped": [list(s) for s in self.skipped],
                "group": self.group}


def is_mechanical(node: NodeSpec, *, integration: bool = False) -> bool:
    """True for a mechanical leaf step."""
    return node.kind == "mechanical" and not integration


_WORD = re.compile(r"[a-z0-9]+")


def is_risky(node: NodeSpec, config: LadderConfig) -> bool:
    """True for a one-way step, or when its goal, acceptance or scope mentions code that deletes, kills or sends data.

    The planner can mark a step one-way to raise its risk; it can't lower it,
    since the risky words are checked whatever the plan says.
    """
    if node.door == "one-way":
        return True
    text = " ".join([node.goal, *node.acceptance, *node.scope]).lower()
    words = set(_WORD.findall(text))
    return any(w in words for w in config.risky_words)


def is_checkable(node: NodeSpec, config: LadderConfig) -> bool:
    """True when Mullet's own checks can judge the step: acceptance commands and a narrow scope."""
    broad = any(g.strip() in ("**", "*", "**/*") for g in node.scope)
    return bool(node.acceptance) and not broad and len(node.scope) <= config.narrow_globs


def step_group(node: NodeSpec, config: LadderConfig, *, integration: bool = False) -> tuple[str, str]:
    """(group, why): mechanical, checkable (a design step its checks can judge) or claude-first."""
    if integration:
        return CLAUDE_FIRST, "a parent's integration fix"
    if node.door == "one-way":
        return CLAUDE_FIRST, "a one-way step (hard to walk back)"
    if node.kind == "mechanical":
        return MECHANICAL, "a mechanical step"
    if node.kind == "design" and is_checkable(node, config) and not is_risky(node, config):
        return CHECKABLE, "a design step with acceptance commands and a narrow scope"
    if node.kind == "design" and is_risky(node, config):
        return CLAUDE_FIRST, "a design step that deletes, kills or sends data"
    if node.kind == "design":
        return CLAUDE_FIRST, (f"a design step without acceptance commands or with more than "
                              f"{config.narrow_globs} narrow scope globs")
    return CLAUDE_FIRST, f"an {node.kind} step"


def step_order(level: str, group: str, *, night: bool = False) -> tuple[str, ...]:
    """The order of rungs for a budget level and step group (``ORDERS``).

    At night a checkable step at the medium level tries local first; every
    other order is the same by day and by night.
    """
    level = level if level in ORDERS else "medium"
    order = ORDERS[level][group]
    if night and group == CHECKABLE and level == "medium":
        return ("local", *order)
    return order


def plan_rungs(node: NodeSpec, profile: Profile, budget: BudgetState, machine: MachineState,
               spend: SpendState, *, config: LadderConfig, public: PublicCheck | None = None,
               integration: bool = False, has_key: Callable[[Rung], bool] = rung_has_key,
               night: bool = False) -> RungChoice:
    """The balanced policy: the rungs `node` may use, in order.

    The order comes from Claude's budget level and the step's group
    (``ORDERS``): when Claude's week has room it comes before the paid rungs,
    when it is tight it is the last resort. ``ladder.start`` then removes rungs
    cheaper than a kind may use, and each rung must pass its own checks
    (allowed executor, public repo, machine room, spending cap, key).
    `public` is the job's public check (None: not checked, so not public).
    """
    allowed = set(profile.allowed_executors)
    skipped: list[tuple[str, str]] = []
    group, why = step_group(node, config, integration=integration)
    if profile.ladder == "claude-only":
        group, why = CLAUDE_FIRST, "the profile says ladder: claude-only"
        order: tuple[str, ...] = ("claude",)
    else:
        order = step_order(budget.level, group, night=night)
    floor = RUNG_ORDER.index(config.start.get(node.kind, "free"))
    configured = {name: [r for r in config.rungs() if r.name == name] for name in RUNG_ORDER}
    rungs: list[Rung] = []
    capped = False
    rung_limited = False  # a paid rung skipped for its own limit: only a cap when no paid rung is left
    for name in RUNG_ORDER:
        if name not in order:
            for rung in configured[name]:
                skipped.append((rung.label(), f"not used for {why} while Claude's budget is {budget.level}"))
    for name in order:
        for rung in configured[name]:
            reason = None
            if rung.executor not in allowed:
                reason = f"the profile doesn't allow the {rung.executor} executor"
            elif rung.name != "claude" and RUNG_ORDER.index(rung.name) < floor:
                reason = f"{node.kind} steps start at {RUNG_ORDER[floor]} (ladder.start)"
            elif rung.name == "free" and not (public and public.public):
                reason = f"not a verified public repo: {public.reason if public else 'not checked'}"
            elif rung.name == "local" and not (night or config.local_by_day):
                reason = "local runs only in night mode (ladder.local_by_day: false)"
            elif rung.name == "local" and not machine.local_ok:
                reason = machine.reason or "the machine has no room for local work"
            elif rung.paid and spend.capped:
                reason = spend.reason or f"the job's spending cap is reached (${spend.spent:.2f} of ${spend.cap:.2f})"
                capped = True
            elif rung.paid and spend.rung_left.get(rung.name, 1.0) <= 0:
                reason = spend.rung_reasons.get(rung.name) or f"the {rung.name} spending limit is reached"
                rung_limited = True
            elif not has_key(rung):
                reason = "no API key for its provider"
            if reason:
                skipped.append((rung.label(), reason))
            else:
                rungs.append(rung)
    capped = capped or (rung_limited and not any(r.paid for r in rungs))
    return RungChoice(rungs, skipped, group, capped)


def choose_rungs(node: NodeSpec, profile: Profile, budget_state: BudgetState, machine_state: MachineState,
                 spend_state: SpendState, *, config: LadderConfig | None = None, public: PublicCheck | None = None,
                 integration: bool = False, has_key: Callable[[Rung], bool] = rung_has_key) -> list[Rung]:
    """The rungs `node` may use, in order: ``plan_rungs`` without the reasons."""
    return plan_rungs(node, profile, budget_state, machine_state, spend_state, config=config or load_config(),
                      public=public, integration=integration, has_key=has_key).rungs


def model_family(model: str | None) -> str:
    """A model's identity across providers: `openrouter/moonshotai/kimi-k2.7-code:free` and
    `moonshotai/kimi-k2.7-code` are the same model."""
    name = (model or "").lower()
    if name.startswith("openrouter/"):
        name = name[len("openrouter/"):]
    return name[:-len(":free")] if name.endswith(":free") else name


def claude_tries(budget: BudgetState, max_attempts: int, used_below: int) -> int:
    """Counted attempts Claude gets at a step: one when its week is tight (the last
    resort), otherwise what is left of the job's attempts after the cheaper rungs, at least one."""
    if budget.level == "high":
        return 1
    return max(1, max_attempts - used_below)


last_rung_tries = claude_tries  # the earlier name


@dataclass(frozen=True)
class Tried:
    """One earlier attempt at a step, as the rung choice sees it."""

    rung_key: str
    rung: str
    model: str | None
    counted: bool  # passed, unverified or failed: used up an attempt
    failed: bool  # failed its gates
    moved_up: bool  # a provider error or broken run: the rung is used, the attempt isn't


@dataclass
class NextRung:
    """The rung for a step's next attempt, or None with the reason."""

    rung: Rung | None
    final: bool = True  # no attempt would follow this one if it failed
    reason: str = ""
    skipped: list[tuple[str, str]] = field(default_factory=list)


def next_rung(choice: RungChoice, tried: list[Tried], budget: BudgetState, *, max_attempts: int,
              config: LadderConfig, extra_claude_tries: int = 0) -> NextRung:
    """One attempt per rung, in the choice's order; Claude gets `claude_tries` (plus `extra_claude_tries`).

    A rung whose model already failed this step's gates elsewhere is never
    tried (only a provider error lets the same model run again on another
    provider). ``config.max_below_claude`` caps the counted attempts on every
    rung but Claude, before and after Claude's own attempts alike: once that
    many have run, only Claude may take the step, and without Claude (or with
    its tries used) the step has run out of rungs. Counted means passed,
    unverified or failed (``Tried.counted``); a provider error, a moved-up or
    an aborted attempt never uses up the allowance. Night mode raises the cap
    through ``apply_night``.
    """
    counted: dict[str, int] = {}
    used: set[str] = set()
    failed_models: set[str] = set()
    for t in tried:
        if t.counted:
            used.add(t.rung_key)
            counted[t.rung_key] = counted.get(t.rung_key, 0) + 1
            if t.failed and t.rung != "claude" and t.model:
                failed_models.add(model_family(t.model))
        elif t.moved_up:
            used.add(t.rung_key)
    claude = next((r for r in choice.rungs if r.name == "claude"), None)
    claude_done = counted.get("claude:", 0)
    below = sum(n for k, n in counted.items() if k != "claude:")
    tries = claude_tries(budget, max_attempts, below) + extra_claude_tries
    cap = config.max_below_claude
    skipped: list[tuple[str, str]] = []

    def usable(rung: Rung, note: bool, below_now: int) -> bool:
        if rung.name == "claude":
            return claude_done < tries
        if rung.key in used:
            return False
        if model_family(rung.model) in failed_models:
            if note:
                skipped.append((rung.label(), "its model already failed this step's checks"))
            return False
        if below_now >= cap:
            if note:
                turn = "Claude's turn" if claude is not None and claude_done < tries else "no more below Claude"
                skipped.append((rung.label(), f"{cap} attempts below Claude already (ladder.max_below_claude); "
                                              f"{turn}"))
            return False
        return True

    for i, rung in enumerate(choice.rungs):
        if not usable(rung, True, below):
            continue
        below_after = below + (0 if rung.name == "claude" else 1)
        later = [r for j, r in enumerate(choice.rungs)
                 if j > i and r.name != "claude" and usable(r, False, below_after)]
        claude_left = claude is not None and claude_done + (1 if rung.name == "claude" else 0) < tries
        return NextRung(rung, final=not later and not claude_left, skipped=skipped)
    total = sum(counted.values())
    return NextRung(None, True, f"ran out of rungs: every rung of the ladder was tried ({total} attempts)", skipped)


# --- Claude's budget level, kept between runs --------------------------------------------


def _level_path() -> Path:
    from mullet.core.paths import state_dir

    return state_dir() / "ladder" / "budget-level.json"


def remember_level(state: BudgetState, now: float) -> None:
    """Keep a known budget level for when there are no fresh figures."""
    if state.weekly_pressure is None:
        return
    path = _level_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"level": state.level, "pressure": state.weekly_pressure, "ts": now}))
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("ladder: can't keep the budget level: %s", exc)


def last_known_level(now: float, max_age: float = LEVEL_MAX_AGE_S) -> BudgetState | None:
    """The last known budget level, if it is under `max_age` old."""
    try:
        data = json.loads(_level_path().read_text())
        if data.get("level") in ORDERS and 0 <= now - float(data["ts"]) < max_age:
            return BudgetState(str(data["level"]), float(data["pressure"]), stale_s=now - float(data["ts"]))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


@dataclass
class Ladder:
    """What the runner asks the ladder, with every outside lookup replaceable for tests."""

    config: LadderConfig = field(default_factory=load_config)
    usage: Callable[[], Any] = current_usage
    machine: Callable[..., MachineState] = machine_state
    public_check: Callable[[Path, str, Profile], PublicCheck] = check_public
    has_key: Callable[[Rung], bool] = rung_has_key
    now: Callable[[], float] = time.time
    public_max_age: float = PUBLIC_CACHE_S
    night: Callable[[], NightState] = night.check

    def budget(self) -> BudgetState:
        """Claude's budget level now; without fresh figures, the last known one (under 6 hours), else medium."""
        state = budget_state(self.config, self.usage())
        if state.weekly_pressure is not None:
            remember_level(state, self.now())
            return state
        return last_known_level(self.now()) or state

    def public(self, repo: Path, base_sha: str, profile: Profile) -> PublicCheck:
        """The public check, cached per repo and base for an hour."""
        return cached_public_check(repo, base_sha, profile, check=self.public_check, now=self.now,
                                   max_age=self.public_max_age)
