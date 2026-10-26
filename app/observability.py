"""
Logging that says which case a line belongs to.

The pipeline runs one case at a time on a worker thread while the API answers requests on others, so a
plain log interleaves lines from different cases with nothing to tell them apart. Here the case id (and,
for web requests, a request id) travels in context variables. `bind` sets them for a block of code, and a
filter on the log handler copies them onto every record, so the code that logs does not have to pass an id
around. Context variables are per thread of execution, so the worker's case id never leaks into a request.

Two formats, chosen by MEDREASON_LOG_FORMAT. "text" is for reading in a terminal. "json" writes one object
per line (time, level, logger, message, case_id, request_id, plus any `extra=` fields such as stage and
seconds), which is what a log collector in a container setup reads.

Only the "medreason" logger tree is configured, with its own handler and propagate off. Without a handler the
standard library prints nothing below WARNING, which is why the runner's INFO lines used to be invisible
under uvicorn. Uvicorn's own loggers are left alone.
"""
from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Iterator, TextIO

_case_id: ContextVar[str | None] = ContextVar("case_id", default=None)
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

ROOT_LOGGER = "medreason"
TEXT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [case=%(case_id)s req=%(request_id)s] %(message)s"

# Attributes every LogRecord has. Anything else on a record came from `extra=` and goes into the JSON line.
_STANDARD = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime", "case_id", "request_id"}


@contextmanager
def bind(case_id: str | None = None, request_id: str | None = None) -> Iterator[None]:
    """Tag every log line written inside the block. Values left as None keep whatever is already bound."""
    tokens = []
    if case_id is not None:
        tokens.append((_case_id, _case_id.set(case_id)))
    if request_id is not None:
        tokens.append((_request_id, _request_id.set(request_id)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def current_case_id() -> str | None:
    return _case_id.get()


class ContextFilter(logging.Filter):
    """Copies the bound ids onto the record. Attached to the handler, not to a logger, because a logger's
    filter is skipped for records that come from its child loggers."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.case_id = _case_id.get() or "-"
        record.request_id = _request_id.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "case_id": getattr(record, "case_id", None),
            "request_id": getattr(record, "request_id", None),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD and not key.startswith("_"):
                out[key] = value
        if record.exc_info:
            out["exception"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure_logging(level: str = "INFO", fmt: str = "text", stream: TextIO | None = None) -> logging.Handler:
    """Install the handler on the medreason logger. Safe to call again: it replaces its own earlier handler."""
    logger = logging.getLogger(ROOT_LOGGER)
    for old in [h for h in logger.handlers if getattr(h, "_medreason", False)]:
        logger.removeHandler(old)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler._medreason = True                                     # type: ignore[attr-defined]
    handler.addFilter(ContextFilter())
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(TEXT_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    return handler


def reset_logging() -> None:
    """Undo configure_logging. Tests use it so one test's handler does not stay on for the next."""
    logger = logging.getLogger(ROOT_LOGGER)
    for old in [h for h in logger.handlers if getattr(h, "_medreason", False)]:
        logger.removeHandler(old)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
