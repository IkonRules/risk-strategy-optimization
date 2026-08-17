# log_config.py
"""
Central logging configuration for the Risk-like data generator.

This module is self-contained: you can choose logging options directly here
(by editing MODULE_DEFAULTS / DEBUG_SWITCHES / DEFAULT_GATE), without needing
to wire anything in your entrypoint unless you want overrides.

Goals:
- One place to turn debug on/off per subsystem (battle_graph, sampler, query, rollout, ranking, etc.)
- No more sprinkling debug=True through call chains
- Optional: write logs to a file while keeping console clean
- Optional: limit noisy debug by state_id/scen/step via filter hooks

Usage (optional in your entrypoint, overrides module defaults):
    from project_risk.infrastructure.log_config import setup_logging, set_debug_switches, ContextGate

    set_debug_switches({
        "runner": True,
        "rollout": True,
        "battle_graph": True,
        "sampler": False,
        "query": False,
        "ranking": False,
    })

    setup_logging(
        level="INFO",
        log_file="run.log",          # or None
        console_level="INFO",
        file_level="DEBUG",
        gate=ContextGate(state_ids={0,1,2}, steps={2}),
    )

Then in modules:
    import logging
    from project_risk.infrastructure.log_config import get_logger

    log = get_logger("risk.battle_graph", state_id=state_id, scen=scen, step=step)
    log.debug("rebuilt graph nodes=%d edges=%d", n, e)

Recommended logger names:
    risk.runner
    risk.rollout
    risk.battle_graph
    risk.partition
    risk.ranking
    risk.query
    risk.sampler
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

# =============================================================================
# MODULE DEFAULTS (edit these here to control behavior without touching callers)
# =============================================================================

MODULE_DEFAULTS: Dict[str, Any] = {
    # Root level: keep at INFO; DEBUG gating is handled by RiskSubsystemFilter.
    "level": "INFO",
    # Console handler verbosity
    "console_level": "INFO",
    # File handler verbosity (if enabled)
    "file_level": "DEBUG",
    # Where to write logs; set to None to disable file logging.
    # You can also set env var RISK_LOG_FILE to override at runtime.
    "log_file": None,
    # If True, setup_logging will be auto-called once when the module is imported.
    # Keep False if you prefer explicit setup in your entrypoint.
    "auto_setup_on_import": False,
    # If True, force reconfigure even if already configured.
    "force_reconfigure": False,
}

# Environment override (optional)
_env_log_file = os.getenv("RISK_LOG_FILE")
if _env_log_file:
    MODULE_DEFAULTS["log_file"] = _env_log_file

# -----------------------------
# Switchboard (edit at runtime)
# -----------------------------
DEBUG_SWITCHES: Dict[str, bool] = {
    # High-level progress / summary
    "runner": True,
    # Multi-step rollout mechanics
    "rollout": False,
    # Battle graph rebuild and semantics
    "battle_graph": False,
    # Partitioning / exact cover / fallback
    "partition": False,
    # Ranking and lookahead
    "ranking": False,
    # Library query + canonicalization
    "query": False,
    # Sampling + distribution collection
    "sampler": False,
}

# Optional: default context gating (edit here for persistent filtering)
DEFAULT_GATE: Optional["ContextGate"] = None

# Internal guard to avoid repeated setup
_CONFIGURED_ONCE = False


def set_debug_switches(updates: Dict[str, bool]) -> None:
    """
    Update DEBUG_SWITCHES in-place.
    Unknown keys are allowed (you might add subsystems later).
    """
    for k, v in (updates or {}).items():
        DEBUG_SWITCHES[str(k)] = bool(v)


def set_module_defaults(updates: Dict[str, Any]) -> None:
    """
    Update MODULE_DEFAULTS in-place.
    Useful if you want to change defaults at runtime without editing the file.
    """
    for k, v in (updates or {}).items():
        MODULE_DEFAULTS[str(k)] = v


def set_default_gate(gate: Optional["ContextGate"]) -> None:
    """Set a module-level default ContextGate used when setup_logging(gate=None)."""
    global DEFAULT_GATE
    DEFAULT_GATE = gate


# -----------------------------
# Optional: contextual filtering
# -----------------------------
@dataclass
class ContextGate:
    """
    Optional debug gating by context keys.
    If you never set these, everything passes (subject to DEBUG_SWITCHES + log levels).

    Example:
        gate = ContextGate(state_ids={0,1,2}, steps={2})
        setup_logging(..., gate=gate)
    """

    state_ids: Optional[set[int]] = None
    scens: Optional[set[int]] = None
    steps: Optional[set[int]] = None

    def allow(self, extra: Dict[str, Any]) -> bool:
        # If no constraints, allow
        if not (self.state_ids or self.scens or self.steps):
            return True

        def _get_int(key: str) -> Optional[int]:
            v = extra.get(key, None)
            try:
                return int(v) if v is not None and v != "" else None
            except Exception:
                return None

        sid = _get_int("state_id")
        scen = _get_int("scen")
        step = _get_int("step")

        if self.state_ids is not None and sid is not None and sid not in self.state_ids:
            return False
        if self.scens is not None and scen is not None and scen not in self.scens:
            return False
        if self.steps is not None and step is not None and step not in self.steps:
            return False

        return True


class RiskSubsystemFilter(logging.Filter):
    """
    Filter that:
      1) maps logger name 'risk.<subsystem>' to DEBUG_SWITCHES[subsystem]
      2) optionally gates DEBUG records by ContextGate

    Rule:
      - INFO/WARN/ERROR always pass
      - DEBUG passes only if subsystem enabled and (optional) context gate passes

    Notes:
      - This filter intentionally only blocks DEBUG (not INFO+).
      - If a logger is not under 'risk.', subsystem becomes 'unknown'
        and is controlled by DEBUG_SWITCHES.get('unknown', False).
    """

    def __init__(self, gate: Optional[ContextGate] = None) -> None:
        super().__init__()
        self.gate = gate

    def filter(self, record: logging.LogRecord) -> bool:
        # Always let INFO+ through
        if record.levelno >= logging.INFO:
            return True

        # DEBUG: require subsystem enabled
        name = record.name or ""
        subsystem = "unknown"
        if name.startswith("risk."):
            # "risk.<subsystem>[.<child>...]" -> we take first segment after "risk."
            tail = name.split(".", 1)[1] if "." in name else ""
            subsystem = tail.split(".", 1)[0] if tail else "unknown"

        if not DEBUG_SWITCHES.get(subsystem, False):
            return False

        # Optional context gating (state_id/scen/step)
        if self.gate is not None:
            extra = {
                "state_id": getattr(record, "state_id", None),
                "scen": getattr(record, "scen", None),
                "step": getattr(record, "step", None),
            }
            if not self.gate.allow(extra):
                return False

        return True


# -----------------------------
# Formatting
# -----------------------------
class SafeExtraFormatter(logging.Formatter):
    """
    Formatter that won't crash if extra fields aren't present.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Ensure these attrs exist so format string works
        for attr in ("state_id", "scen", "step"):
            if not hasattr(record, attr):
                setattr(record, attr, "")
        return super().format(record)

class ConditionalFormatter(logging.Formatter):
    """Formatter that can omit context fields for selected loggers.

    By default we include state_id/scen/step in the format. For loggers like
    'risk.test' (and any others you add), the extra context is usually noise,
    so we switch to a simpler format.
    """

    def __init__(
        self,
        detailed_fmt: str,
        simple_fmt: str,
        datefmt: str | None = None,
        simple_prefixes: tuple[str, ...] = ("risk.test",),
    ) -> None:
        super().__init__(detailed_fmt, datefmt=datefmt)
        self._detailed = SafeExtraFormatter(detailed_fmt, datefmt=datefmt)
        self._simple = logging.Formatter(simple_fmt, datefmt=datefmt)
        self._simple_prefixes = tuple(simple_prefixes)

    def format(self, record: logging.LogRecord) -> str:
        name = record.name or ""
        for p in self._simple_prefixes:
            if name == p or name.startswith(p + "."):
                return self._simple.format(record)
        return self._detailed.format(record)



DEFAULT_FMT = "%(asctime)s %(levelname)s %(name)s state=%(state_id)s scen=%(scen)s step=%(step)s | %(message)s"
SIMPLE_FMT = "%(asctime)s %(levelname)s %(name)s | %(message)s"
DEFAULT_DATEFMT = "%H:%M:%S"


def setup_logging(
    *,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console_level: Optional[str] = None,
    file_level: Optional[str] = None,
    gate: Optional[ContextGate] = None,
    force: Optional[bool] = None,
) -> None:
    """
    Configure root logging.

    If arguments are None, this uses MODULE_DEFAULTS.

    - level: root level (keep at INFO; DEBUG gating is handled by filter)
    - log_file: if provided, also log to this file
    - console_level: level for console handler (INFO recommended)
    - file_level: level for file handler (DEBUG recommended)
    - gate: optional ContextGate for only debugging certain state/scen/step slices
    - force: if True, always reconfigure even if setup already ran once
    """
    global _CONFIGURED_ONCE

    if force is None:
        force = bool(MODULE_DEFAULTS.get("force_reconfigure", False))

    if _CONFIGURED_ONCE and not force:
        return

    # Resolve defaults
    if level is None:
        level = str(MODULE_DEFAULTS.get("level", "INFO"))
    if console_level is None:
        console_level = str(MODULE_DEFAULTS.get("console_level", "INFO"))
    if file_level is None:
        file_level = str(MODULE_DEFAULTS.get("file_level", "DEBUG"))
    if log_file is None:
        log_file = MODULE_DEFAULTS.get("log_file", None)
    if gate is None:
        gate = DEFAULT_GATE

    root = logging.getLogger()

    # Clear handlers to avoid duplicate output when reconfiguring
    root.handlers.clear()
    root.setLevel(_parse_level(level))

    filt = RiskSubsystemFilter(gate=gate)
    formatter = ConditionalFormatter(DEFAULT_FMT, SIMPLE_FMT, datefmt=DEFAULT_DATEFMT, simple_prefixes=("risk.test",))

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(_parse_level(console_level))
    ch.addFilter(filt)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler (optional)
    if log_file:
        fh = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
        fh.setLevel(_parse_level(file_level))
        fh.addFilter(filt)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Keep third-party noise down unless you explicitly enable it
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("networkx").setLevel(logging.WARNING)

    _CONFIGURED_ONCE = True


def _parse_level(s: str) -> int:
    s2 = (s or "INFO").upper().strip()
    return {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }.get(s2, logging.INFO)


# -----------------------------
# Convenience: logger adapters
# -----------------------------
def get_logger(
    name: str,
    *,
    state_id: Optional[int] = None,
    scen: Optional[int] = None,
    step: Optional[int] = None,
):
    """
    Optional helper to inject context without repeating `extra=...` everywhere.

    Example:
        log = get_logger("risk.battle_graph", state_id=state_id, scen=scen, step=step)
        log.debug("rebuilt graph nodes=%d edges=%d", n, e)
    """
    base = logging.getLogger(name)
    extra = {"state_id": state_id, "scen": scen, "step": step}
    return logging.LoggerAdapter(base, extra)


# =============================================================================
# Auto-setup (optional)
# =============================================================================
if bool(MODULE_DEFAULTS.get("auto_setup_on_import", False)):
    setup_logging()
