import json
import logging
from datetime import UTC, datetime
from typing import Any

_CONFIGURED = False


class JSONFormatter(logging.Formatter):
    """Renders each log record as a single JSON line.

    CloudWatch Logs Insights and local `jq` both work better against structured
    JSON than free-text messages, so this is the only formatter used anywhere in the
    project — `print()` is never an acceptable substitute for logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    """Idempotently configures the root logger to emit JSON to stdout.

    Safe to call multiple times (e.g. once per Lambda module import plus once in a
    script's `if __name__ == "__main__"` block) — only the first call takes effect.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    _CONFIGURED = True
