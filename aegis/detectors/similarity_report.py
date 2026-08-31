"""
Unified Similarity Report Generator -- AEGIS v4.0.

Merges n-gram, semantic, and self-plagiarism match results into a single
annotated document view with:
  - Character-level match highlighting (spans tagged with source + score)
  - Source attribution (matched excerpt side-by-side)
  - Similarity percentage (matched chars / total body chars)
  - Color-coded severity (verbatim / paraphrase / semantic)

This module does NOT run detectors itself — it consumes their output and
produces a consolidated, front-end-ready representation.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MatchSpan:
    """A single highlighted span in the submission document."""
    start_offset: int       # char offset in document body
    end_offset: int         # char offset end (exclusive)
    matched_text: str       # the text in the submission
    source_label: str       # corpus document label
    source_excerpt: str     # matched text from the source
    similarity_score: float # 0.0–1.0
    match_type: str         # "verbatim" | "paraphrase" | "semantic"
    detector: str           # "ngram" | "semantic" | "self_plagiarism"
    color: str = ""         # CSS color for the frontend

    def __post_init__(self):
        if not self.color:
            self.color = {
                "verbatim": "#ef4444",     # red
                "paraphrase": "#f97316",   # orange
                "semantic": "#eab308",     # yellow
            }.get(self.match_type, "#94a3b8")


@dataclass
class SimilarityReport:
    """Complete similarity analysis for a document."""
    submission_path: str
    body_text_length: int           # total chars in body text
    matched_char_count: int = 0     # chars covered by at least one match
    similarity_percentage: float = 0.0  # matched_char_count / body_text_length
    spans: list[MatchSpan] = field(default_factory=list)

    # Per-detector breakdown
    verbatim_match_count: int = 0
    paraphrase_match_count: int = 0
    semantic_match_count: int = 0

    # Source-level summary: how much of each source was matched
    source_breakdown: dict[str, dict] = field(default_factory=dict)

    @property
    def has_matches(self) -> bool:
        return len(self.spans) > 0

    def to_dict(self) -> dict:
        return {
            "submission_path": self.submission_path,
            "body_text_length": self.body_text_length,
            "matched_char_count": self.matched_char_count,
            "similarity_percentage": round(self.similarity_percentage, 2),
            "verbatim_match_count": self.verbatim_match_count,
            "paraphrase_match_count": self.paraphrase_match_count,
            "semantic_match_count": self.semantic_match_count,
            "source_breakdown": self.source_breakdown,
            "spans": [
                {
                    "start": s.start_offset,
                    "end": s.end_offset,
                    "text": s.matched_text[:200],
                    "source_label": s.source_label,
                    "source_excerpt": s.source_excerpt[:200],
                    "score": round(s.similarity_score, 3),
                    "type": s.match_type,
                    "detector": s.detector,
                    "color": s.color,
                }
                for s in self.spans
            ],
        }


class SimilarityReportGenerator:
    """
    Consolidates match results from n-gram, semantic, and self-plagiarism
    detectors into a unified SimilarityReport with character-level spans.

    Usage::

        from aegis.detectors.similarity_report import SimilarityReportGenerator

        gen = SimilarityReportGenerator()
        report = gen.generate(
            body_text=parsed_doc.body_text,
            submission_path="paper.docx",
            ngram_matches=analysis.ngram_matches,
            semantic_matches=analysis.semantic_matches,
            self_plagiarism_result=analysis.self_plagiarism_result,
        )
        print(f"Similarity: {report.similarity_percentage:.1f}%")
    """

    def __init__(self, verbatim_threshold: float = 0.50,
                 paraphrase_threshold: float = 0.30):
        """
        verbatim_threshold: n-gram Jaccard above which = verbatim match
        paraphrase_threshold: n-gram Jaccard above which = paraphrase
        """
        self._verbatim_threshold = verbatim_threshold
        self._paraphrase_threshold = paraphrase_threshold

    def generate(
        self,
        body_text: str,
        submission_path: str,
        ngram_matches: list = None,
        semantic_matches: list = None,
        self_plagiarism_result=None,
    ) -> SimilarityReport:
        """Generate a consolidated similarity report."""
        ngram_matches = ngram_matches or []
        semantic_matches = semantic_matches or []

        report = SimilarityReport(
            submission_path=submission_path,
            body_text_length=len(body_text),
        )

        all_spans: list[MatchSpan] = []

        # 1. Convert n-gram matches to spans
        for match in ngram_matches:
            spans = self._ngram_to_spans(match, body_text)
            all_spans.extend(spans)

        # 2. Convert semantic matches to spans
        for match in semantic_matches:
            spans = self._semantic_to_spans(match, body_text)
            all_spans.extend(spans)

        # 3. Convert self-plagiarism matches to spans
        if self_plagiarism_result:
            spans = self._self_plag_to_spans(self_plagiarism_result, body_text)
            all_spans.extend(spans)

        # 4. Merge overlapping spans (keep highest score)
        merged = self._merge_spans(all_spans)

        # 5. Calculate coverage
        report.spans = sorted(merged, key=lambda s: s.start_offset)
        report.matched_char_count = self._calculate_coverage(merged)
        if report.body_text_length > 0:
            report.similarity_percentage = round(
                (report.matched_char_count / report.body_text_length) * 100, 2)

        # 6. Count by type
        report.verbatim_match_count = sum(
            1 for s in merged if s.match_type == "verbatim")
        report.paraphrase_match_count = sum(
            1 for s in merged if s.match_type == "paraphrase")
        report.semantic_match_count = sum(
            1 for s in merged if s.match_type == "semantic")

        # 7. Source breakdown
        report.source_breakdown = self._source_breakdown(merged)

        return report

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _ngram_to_spans(self, match, body_text: str) -> list[MatchSpan]:
        """Convert an NGramMatch to character-level MatchSpans."""
        spans = []
        query_text = match.query_segment

        # Find the query segment's position in the body text
        idx = body_text.find(query_text[:100])
        if idx < 0:
            # Try a shorter prefix match
            for prefix_len in [80, 60, 40, 30]:
                idx = body_text.find(query_text[:prefix_len])
                if idx >= 0:
                    break

        if idx >= 0:
            # Determine match type based on Jaccard score
            if match.jaccard_estimate >= self._verbatim_threshold:
                match_type = "verbatim"
            elif match.jaccard_estimate >= self._paraphrase_threshold:
                match_type = "paraphrase"
            else:
                match_type = "semantic"

            span_len = min(len(query_text), len(body_text) - idx)
            spans.append(MatchSpan(
                start_offset=idx,
                end_offset=idx + span_len,
                matched_text=body_text[idx:idx + span_len][:400],
                source_label=match.source_label,
                source_excerpt=match.source_segment[:400],
                similarity_score=match.jaccard_estimate,
                match_type=match_type,
                detector="ngram",
            ))

        return spans

    def _semantic_to_spans(self, match, body_text: str) -> list[MatchSpan]:
        """Convert a SemanticMatch to character-level MatchSpans."""
        spans = []
        query_sent = match.query_sentence

        # Find the query sentence position in body text
        idx = body_text.find(query_sent[:80])
        if idx < 0:
            for prefix_len in [60, 40, 30]:
                idx = body_text.find(query_sent[:prefix_len])
                if idx >= 0:
                    break

        if idx >= 0:
            span_len = min(len(query_sent), len(body_text) - idx)
            spans.append(MatchSpan(
                start_offset=idx,
                end_offset=idx + span_len,
                matched_text=body_text[idx:idx + span_len],
                source_label=match.source_label,
                source_excerpt=match.source_sentence[:400],
                similarity_score=match.cosine_score,
                match_type="semantic" if match.is_paraphrase else "paraphrase",
                detector="semantic",
            ))

        return spans

    def _self_plag_to_spans(self, result, body_text: str) -> list[MatchSpan]:
        """Convert self-plagiarism overlaps to MatchSpans."""
        spans = []
        if not hasattr(result, 'overlapping_segments'):
            return spans

        for seg in result.overlapping_segments:
            query_text = getattr(seg, 'submission_text', '')
            if not query_text:
                continue

            idx = body_text.find(query_text[:80])
            if idx < 0:
                for prefix_len in [60, 40, 30]:
                    idx = body_text.find(query_text[:prefix_len])
                    if idx >= 0:
                        break

            if idx >= 0:
                span_len = min(len(query_text), len(body_text) - idx)
                score = getattr(seg, 'similarity', 0.5)
                spans.append(MatchSpan(
                    start_offset=idx,
                    end_offset=idx + span_len,
                    matched_text=body_text[idx:idx + span_len][:400],
                    source_label=getattr(seg, 'prior_work_label', 'prior_work'),
                    source_excerpt=getattr(seg, 'prior_text', '')[:400],
                    similarity_score=score,
                    match_type="verbatim" if score > 0.8 else "paraphrase",
                    detector="self_plagiarism",
                ))

        return spans

    # ------------------------------------------------------------------
    # Span merging and coverage
    # ------------------------------------------------------------------

    def _merge_spans(self, spans: list[MatchSpan]) -> list[MatchSpan]:
        """
        Merge overlapping spans. When spans overlap, keep the one with
        the higher similarity score. Different match types can coexist
        if they don't overlap.
        """
        if not spans:
            return []

        # Sort by start offset
        sorted_spans = sorted(spans, key=lambda s: (s.start_offset, -s.similarity_score))
        merged: list[MatchSpan] = [sorted_spans[0]]

        for span in sorted_spans[1:]:
            last = merged[-1]
            # Check overlap
            if span.start_offset < last.end_offset:
                # Overlapping: keep higher score
                if span.similarity_score > last.similarity_score:
                    # Extend or replace
                    merged[-1] = span
                # else: keep existing (higher or equal score)
            else:
                merged.append(span)

        return merged

    def _calculate_coverage(self, spans: list[MatchSpan]) -> int:
        """Calculate total unique characters covered by match spans."""
        if not spans:
            return 0

        # Merge overlapping intervals to count unique characters
        intervals = sorted([(s.start_offset, s.end_offset) for s in spans])
        merged_intervals = [intervals[0]]

        for start, end in intervals[1:]:
            last_start, last_end = merged_intervals[-1]
            if start <= last_end:
                merged_intervals[-1] = (last_start, max(last_end, end))
            else:
                merged_intervals.append((start, end))

        return sum(end - start for start, end in merged_intervals)

    def _source_breakdown(self, spans: list[MatchSpan]) -> dict[str, dict]:
        """Group matched spans by source and compute per-source stats."""
        sources: dict[str, dict] = {}
        for span in spans:
            label = span.source_label
            if label not in sources:
                sources[label] = {
                    "match_count": 0,
                    "total_chars": 0,
                    "max_score": 0.0,
                    "match_types": set(),
                }
            sources[label]["match_count"] += 1
            sources[label]["total_chars"] += span.end_offset - span.start_offset
            sources[label]["max_score"] = max(
                sources[label]["max_score"], span.similarity_score)
            sources[label]["match_types"].add(span.match_type)

        # Convert sets to lists for JSON serialization
        for label in sources:
            sources[label]["match_types"] = sorted(sources[label]["match_types"])
            sources[label]["max_score"] = round(sources[label]["max_score"], 3)

        return sources
