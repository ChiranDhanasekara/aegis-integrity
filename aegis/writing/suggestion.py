"""
WritingSuggestion data model and SuggestionSet management.

Each suggestion is a self-contained, reviewable unit: it identifies a
span of text, proposes a replacement, explains why, and tracks whether
the user has accepted, rejected, or modified it.

SuggestionSet is an ordered collection that handles:
  - Deduplication (overlapping suggestions for the same span)
  - Conflict resolution (two suggestions modifying the same text)
  - Bulk accept/reject by category
  - Export to JSON for the web editor
"""

from __future__ import annotations
import uuid
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# Valid categories for writing suggestions
CATEGORIES = frozenset({
    "clarity",        # Sentence restructuring for readability
    "grammar",        # Mechanical grammar correction
    "style",          # Academic tone, contractions, formality
    "spelling",       # Spelling corrections (incl. UK/US consistency)
    "wordiness",      # Verbose phrases that can be shortened
    "passive_voice",  # Passive → active voice conversion
    "hedge",          # Excessive hedging language
    "nominalization", # Abstract nouns → concrete verbs
    "repetition",     # Repeated words or structures
    "sentence_length",# Overly long sentences
})

SEVERITIES = frozenset({"info", "warning", "error"})
STATUSES = frozenset({"pending", "accepted", "rejected", "modified"})


@dataclass
class WritingSuggestion:
    """A single actionable writing suggestion with character offsets."""

    category: str           # One of CATEGORIES
    severity: str           # "info" | "warning" | "error"
    original_text: str      # The exact text span being flagged
    suggested_text: str     # Proposed replacement text
    explanation: str        # Human-readable rationale
    start_offset: int       # Character offset in document (0-based)
    end_offset: int         # Character offset end (exclusive)
    confidence: float       # 0.0 – 1.0; how certain the suggestion is correct

    # Tracking state
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "pending"  # "pending" | "accepted" | "rejected" | "modified"
    modified_text: Optional[str] = None  # User's custom edit (when status="modified")

    # Optional metadata
    rule_source: Optional[str] = None   # Citation for the rule being applied
    paragraph_index: Optional[int] = None
    sentence_index: Optional[int] = None

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(
                f"Invalid category '{self.category}'; must be one of {sorted(CATEGORIES)}"
            )
        if self.severity not in SEVERITIES:
            raise ValueError(
                f"Invalid severity '{self.severity}'; must be one of {sorted(SEVERITIES)}"
            )
        if self.status not in STATUSES:
            raise ValueError(
                f"Invalid status '{self.status}'; must be one of {sorted(STATUSES)}"
            )
        self.confidence = max(0.0, min(1.0, self.confidence))

    def accept(self) -> None:
        """Mark this suggestion as accepted."""
        self.status = "accepted"

    def reject(self) -> None:
        """Mark this suggestion as rejected."""
        self.status = "rejected"

    def modify(self, custom_text: str) -> None:
        """Accept with a user-modified replacement."""
        self.status = "modified"
        self.modified_text = custom_text

    @property
    def final_text(self) -> str:
        """The text that should replace the original when applied."""
        if self.status == "modified" and self.modified_text is not None:
            return self.modified_text
        return self.suggested_text

    @property
    def is_resolved(self) -> bool:
        return self.status != "pending"

    def to_dict(self) -> dict:
        """Serialize for JSON / web editor consumption."""
        d = asdict(self)
        d["final_text"] = self.final_text
        d["is_resolved"] = self.is_resolved
        return d


class SuggestionSet:
    """
    An ordered collection of WritingSuggestions for a single document.

    Handles deduplication, conflict resolution, and batch operations.
    """

    def __init__(self):
        self._suggestions: list[WritingSuggestion] = []
        self._by_id: dict[str, WritingSuggestion] = {}

    def add(self, suggestion: WritingSuggestion) -> bool:
        """
        Add a suggestion. Returns False if it overlaps with an existing
        higher-confidence suggestion for the same span (deduplicated).
        """
        # Check for overlapping suggestions
        for existing in self._suggestions:
            if self._overlaps(suggestion, existing):
                # Keep the higher-confidence one
                if existing.confidence >= suggestion.confidence:
                    return False
                else:
                    # Replace the existing with the new one
                    self._suggestions.remove(existing)
                    del self._by_id[existing.id]
                    break

        self._suggestions.append(suggestion)
        self._by_id[suggestion.id] = suggestion
        return True

    def add_all(self, suggestions: list[WritingSuggestion]) -> int:
        """Add multiple suggestions. Returns count of suggestions actually added."""
        return sum(1 for s in suggestions if self.add(s))

    def get(self, suggestion_id: str) -> Optional[WritingSuggestion]:
        return self._by_id.get(suggestion_id)

    def accept(self, suggestion_id: str) -> bool:
        s = self._by_id.get(suggestion_id)
        if s:
            s.accept()
            return True
        return False

    def reject(self, suggestion_id: str) -> bool:
        s = self._by_id.get(suggestion_id)
        if s:
            s.reject()
            return True
        return False

    def modify(self, suggestion_id: str, custom_text: str) -> bool:
        s = self._by_id.get(suggestion_id)
        if s:
            s.modify(custom_text)
            return True
        return False

    def accept_category(self, category: str) -> int:
        """Accept all pending suggestions of a given category."""
        count = 0
        for s in self._suggestions:
            if s.category == category and s.status == "pending":
                s.accept()
                count += 1
        return count

    def reject_category(self, category: str) -> int:
        """Reject all pending suggestions of a given category."""
        count = 0
        for s in self._suggestions:
            if s.category == category and s.status == "pending":
                s.reject()
                count += 1
        return count

    @property
    def all(self) -> list[WritingSuggestion]:
        """All suggestions sorted by document position."""
        return sorted(self._suggestions, key=lambda s: s.start_offset)

    @property
    def pending(self) -> list[WritingSuggestion]:
        return [s for s in self.all if s.status == "pending"]

    @property
    def accepted(self) -> list[WritingSuggestion]:
        return [s for s in self.all if s.status in ("accepted", "modified")]

    @property
    def rejected(self) -> list[WritingSuggestion]:
        return [s for s in self.all if s.status == "rejected"]

    def by_category(self, category: str) -> list[WritingSuggestion]:
        return [s for s in self.all if s.category == category]

    @property
    def summary(self) -> dict:
        """Category-level counts for the suggestion sidebar."""
        cats: dict[str, dict] = {}
        for s in self._suggestions:
            if s.category not in cats:
                cats[s.category] = {"total": 0, "pending": 0, "accepted": 0, "rejected": 0}
            cats[s.category]["total"] += 1
            cats[s.category][s.status] += 1
        return {
            "total": len(self._suggestions),
            "pending": len(self.pending),
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "by_category": cats,
        }

    def apply_to_text(self, original_text: str) -> str:
        """
        Apply all accepted/modified suggestions to the original document text.

        Applies in reverse offset order to preserve character positions.
        Skips suggestions whose offsets don't match the original text
        (defensive against stale suggestions).
        """
        result = original_text
        # Sort by start_offset descending so earlier offsets aren't shifted
        for s in sorted(self.accepted, key=lambda x: x.start_offset, reverse=True):
            # Verify the original text still matches at the expected offset
            actual = result[s.start_offset:s.end_offset]
            if actual == s.original_text:
                result = result[:s.start_offset] + s.final_text + result[s.end_offset:]
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {"suggestions": [s.to_dict() for s in self.all],
             "summary": self.summary},
            indent=indent, ensure_ascii=False,
        )

    def __len__(self) -> int:
        return len(self._suggestions)

    def __iter__(self):
        return iter(self.all)

    @staticmethod
    def _overlaps(a: WritingSuggestion, b: WritingSuggestion) -> bool:
        """Check if two suggestions target overlapping character spans."""
        return a.start_offset < b.end_offset and b.start_offset < a.end_offset
