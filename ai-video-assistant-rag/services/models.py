from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str

    @property
    def label(self) -> str:
        return f"{int(self.start // 60):02d}:{int(self.start % 60):02d}"


@dataclass
class MeetingReport:
    summary: list[str] = field(default_factory=list)
    action_items: list[dict[str, str]] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
