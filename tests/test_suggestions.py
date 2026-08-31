"""
Tests for aegis.writing.suggestion -- WritingSuggestion and SuggestionSet.
"""

import json
import pytest
from aegis.writing.suggestion import WritingSuggestion, SuggestionSet


class TestWritingSuggestion:

    def test_create_valid_suggestion(self):
        s = WritingSuggestion(
            category="wordiness",
            severity="info",
            original_text="in order to",
            suggested_text="to",
            explanation="Shorten for conciseness.",
            start_offset=10,
            end_offset=21,
            confidence=0.85,
        )
        assert s.status == "pending"
        assert s.category == "wordiness"
        assert s.confidence == 0.85
        assert len(s.id) == 12  # hex UUID prefix

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="Invalid category"):
            WritingSuggestion(
                category="not_a_category",
                severity="info",
                original_text="x", suggested_text="y",
                explanation="test",
                start_offset=0, end_offset=1, confidence=0.5,
            )

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError, match="Invalid severity"):
            WritingSuggestion(
                category="grammar",
                severity="critical",  # Not a valid severity
                original_text="x", suggested_text="y",
                explanation="test",
                start_offset=0, end_offset=1, confidence=0.5,
            )

    def test_confidence_clamped(self):
        s = WritingSuggestion(
            category="grammar", severity="info",
            original_text="x", suggested_text="y", explanation="t",
            start_offset=0, end_offset=1, confidence=1.5,
        )
        assert s.confidence == 1.0

        s2 = WritingSuggestion(
            category="grammar", severity="info",
            original_text="x", suggested_text="y", explanation="t",
            start_offset=0, end_offset=1, confidence=-0.3,
        )
        assert s2.confidence == 0.0

    def test_accept(self):
        s = WritingSuggestion(
            category="grammar", severity="info",
            original_text="x", suggested_text="y", explanation="t",
            start_offset=0, end_offset=1, confidence=0.5,
        )
        s.accept()
        assert s.status == "accepted"
        assert s.is_resolved
        assert s.final_text == "y"

    def test_reject(self):
        s = WritingSuggestion(
            category="grammar", severity="info",
            original_text="x", suggested_text="y", explanation="t",
            start_offset=0, end_offset=1, confidence=0.5,
        )
        s.reject()
        assert s.status == "rejected"
        assert s.is_resolved

    def test_modify(self):
        s = WritingSuggestion(
            category="grammar", severity="info",
            original_text="x", suggested_text="y", explanation="t",
            start_offset=0, end_offset=1, confidence=0.5,
        )
        s.modify("z")
        assert s.status == "modified"
        assert s.is_resolved
        assert s.final_text == "z"

    def test_to_dict(self):
        s = WritingSuggestion(
            category="wordiness", severity="info",
            original_text="in order to", suggested_text="to",
            explanation="Shorten.", start_offset=5, end_offset=16,
            confidence=0.8,
        )
        d = s.to_dict()
        assert d["category"] == "wordiness"
        assert d["final_text"] == "to"
        assert d["is_resolved"] is False


class TestSuggestionSet:

    def _make(self, start=0, end=5, category="grammar", confidence=0.8):
        return WritingSuggestion(
            category=category, severity="info",
            original_text="x" * (end - start), suggested_text="y",
            explanation="test", start_offset=start, end_offset=end,
            confidence=confidence,
        )

    def test_add_and_len(self):
        ss = SuggestionSet()
        ss.add(self._make(0, 5))
        ss.add(self._make(10, 15))
        assert len(ss) == 2

    def test_deduplication_overlapping(self):
        ss = SuggestionSet()
        s1 = self._make(0, 10, confidence=0.7)
        s2 = self._make(5, 15, confidence=0.9)  # Overlaps, higher confidence
        ss.add(s1)
        added = ss.add(s2)
        assert added is True
        assert len(ss) == 1
        assert ss.all[0].confidence == 0.9

    def test_deduplication_lower_confidence_rejected(self):
        ss = SuggestionSet()
        s1 = self._make(0, 10, confidence=0.9)
        s2 = self._make(5, 15, confidence=0.3)  # Overlaps, lower confidence
        ss.add(s1)
        added = ss.add(s2)
        assert added is False
        assert len(ss) == 1

    def test_non_overlapping_both_kept(self):
        ss = SuggestionSet()
        ss.add(self._make(0, 5))
        ss.add(self._make(20, 30))
        assert len(ss) == 2

    def test_accept_by_id(self):
        ss = SuggestionSet()
        s = self._make(0, 5)
        ss.add(s)
        assert ss.accept(s.id)
        assert s.status == "accepted"

    def test_reject_by_id(self):
        ss = SuggestionSet()
        s = self._make(0, 5)
        ss.add(s)
        assert ss.reject(s.id)
        assert s.status == "rejected"

    def test_modify_by_id(self):
        ss = SuggestionSet()
        s = self._make(0, 5)
        ss.add(s)
        assert ss.modify(s.id, "custom")
        assert s.final_text == "custom"

    def test_accept_category(self):
        ss = SuggestionSet()
        ss.add(self._make(0, 5, category="grammar"))
        ss.add(self._make(10, 15, category="grammar"))
        ss.add(self._make(20, 25, category="wordiness"))
        count = ss.accept_category("grammar")
        assert count == 2
        assert len(ss.accepted) == 2
        assert len(ss.pending) == 1

    def test_reject_category(self):
        ss = SuggestionSet()
        ss.add(self._make(0, 5, category="spelling"))
        ss.add(self._make(10, 15, category="spelling"))
        count = ss.reject_category("spelling")
        assert count == 2

    def test_summary(self):
        ss = SuggestionSet()
        ss.add(self._make(0, 5, category="grammar"))
        ss.add(self._make(10, 15, category="wordiness"))
        summary = ss.summary
        assert summary["total"] == 2
        assert summary["pending"] == 2
        assert "grammar" in summary["by_category"]

    def test_apply_to_text(self):
        text = "We used in order to do this."
        s = WritingSuggestion(
            category="wordiness", severity="info",
            original_text="in order to",
            suggested_text="to",
            explanation="Shorten.",
            start_offset=8, end_offset=19,
            confidence=0.8,
        )
        ss = SuggestionSet()
        ss.add(s)
        s.accept()
        result = ss.apply_to_text(text)
        assert result == "We used to do this."

    def test_apply_multiple_non_overlapping(self):
        text = "We need in order to start. Also in order to finish."
        # First "in order to" at index 8
        idx1 = text.index("in order to")
        s1 = WritingSuggestion(
            category="wordiness", severity="info",
            original_text="in order to",
            suggested_text="to",
            explanation="Shorten.",
            start_offset=idx1, end_offset=idx1 + 11,
            confidence=0.8,
        )
        # Second "in order to" at index 32
        idx2 = text.index("in order to", idx1 + 11)
        s2 = WritingSuggestion(
            category="wordiness", severity="info",
            original_text="in order to",
            suggested_text="to",
            explanation="Shorten.",
            start_offset=idx2, end_offset=idx2 + 11,
            confidence=0.8,
        )
        ss = SuggestionSet()
        ss.add(s1)
        ss.add(s2)
        s1.accept()
        s2.accept()
        result = ss.apply_to_text(text)
        assert "in order to" not in result
        assert result == "We need to start. Also to finish."

    def test_to_json_is_valid(self):
        ss = SuggestionSet()
        ss.add(self._make(0, 5))
        j = ss.to_json()
        data = json.loads(j)
        assert "suggestions" in data
        assert "summary" in data
        assert len(data["suggestions"]) == 1

    def test_sorted_by_offset(self):
        ss = SuggestionSet()
        ss.add(self._make(50, 60))
        ss.add(self._make(0, 10))
        ss.add(self._make(25, 35))
        offsets = [s.start_offset for s in ss.all]
        assert offsets == [0, 25, 50]
