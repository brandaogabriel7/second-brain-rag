"""Collect and summarize errors during ingestion."""

import logging
from dataclasses import dataclass, field

from errors import RecoverableError

logger = logging.getLogger(__name__)


@dataclass
class ErrorCollector:
    """Collects errors during ingestion for summary at end."""

    errors: list[RecoverableError] = field(default_factory=list)

    def add(self, error: RecoverableError) -> None:
        """Add an error to the collection and log it."""
        self.errors.append(error)
        logger.warning(str(error))

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def count(self) -> int:
        return len(self.errors)

    def summarize(self) -> str:
        """Return a human-readable summary of all errors."""
        if not self.errors:
            return "No errors occurred."

        # Group by error type
        by_type: dict[str, list[RecoverableError]] = {}
        for err in self.errors:
            key = type(err).__name__
            by_type.setdefault(key, []).append(err)

        lines = [f"Encountered {len(self.errors)} error(s):"]
        for error_type, errors in by_type.items():
            lines.append(f"  - {error_type}: {len(errors)}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear collected errors."""
        self.errors.clear()
