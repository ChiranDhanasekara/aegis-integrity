"""
Clarity Scorer -- AEGIS Writing Assistant v4.0.

Per-sentence readability analysis and paragraph coherence scoring.
This module provides granular clarity metrics that complement the
document-level statistics already computed by StylometricAnalyzer.

Features:
  - Per-sentence Flesch-Kincaid Grade Level
  - Per-sentence Fog Index
  - Sentence complexity heat map (scored 0–1)
  - Consecutive repetition detection (structure + vocabulary)
  - Paragraph coherence scoring (topic → support → transition flow)
  - Overall clarity score (0–100)
"""

from __future__ import annotations
import re
import math
import logging
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SentenceClarity:
    """Clarity metrics for a single sentence."""
    text: str
    start_offset: int
    end_offset: int
    word_count: int
    syllable_count: int
    fk_grade: float          # Flesch-Kincaid Grade Level
    fog_index: float          # Gunning Fog Index
    complexity_score: float   # 0.0 (simple) – 1.0 (complex)
    has_passive_voice: bool
    nominalization_count: int
    sentence_index: int
    paragraph_index: int


@dataclass
class RepetitionIssue:
    """A detected repetition problem between consecutive sentences."""
    sentence_a_index: int
    sentence_b_index: int
    sentence_a_text: str
    sentence_b_text: str
    repetition_type: str      # "vocabulary" | "structure" | "opening"
    shared_content: str       # The repeated words/pattern
    severity: float           # 0.0–1.0
    start_offset: int
    end_offset: int


@dataclass
class ParagraphCoherence:
    """Coherence metrics for a single paragraph."""
    paragraph_index: int
    text_preview: str
    sentence_count: int
    topic_consistency: float   # 0.0–1.0; how well sentences relate
    transition_score: float    # 0.0–1.0; quality of between-sentence flow
    overall_coherence: float   # 0.0–1.0; combined score
    start_offset: int
    end_offset: int


@dataclass
class ClarityReport:
    """Complete clarity analysis for a document."""
    sentences: list[SentenceClarity] = field(default_factory=list)
    repetitions: list[RepetitionIssue] = field(default_factory=list)
    paragraph_coherence: list[ParagraphCoherence] = field(default_factory=list)

    # Document-level summary
    overall_clarity_score: float = 0.0   # 0–100
    avg_fk_grade: float = 0.0
    avg_fog_index: float = 0.0
    avg_sentence_words: float = 0.0
    complex_sentence_count: int = 0      # sentences with complexity > 0.7
    total_sentences: int = 0
    total_paragraphs: int = 0

    @property
    def complexity_distribution(self) -> dict[str, int]:
        """Bin sentences by complexity level."""
        bins = {"simple": 0, "moderate": 0, "complex": 0, "very_complex": 0}
        for s in self.sentences:
            if s.complexity_score < 0.3:
                bins["simple"] += 1
            elif s.complexity_score < 0.6:
                bins["moderate"] += 1
            elif s.complexity_score < 0.8:
                bins["complex"] += 1
            else:
                bins["very_complex"] += 1
        return bins


# Transition/discourse connectors for coherence scoring
_TRANSITION_WORDS = {
    "however", "therefore", "moreover", "furthermore", "consequently",
    "nevertheless", "additionally", "similarly", "conversely", "meanwhile",
    "nonetheless", "thus", "hence", "accordingly", "likewise", "alternatively",
    "specifically", "notably", "indeed", "subsequently",
}

_TRANSITION_PHRASES = [
    "in addition", "on the other hand", "as a result", "for example",
    "for instance", "in contrast", "in particular", "in other words",
    "as such", "to this end", "in summary", "in conclusion",
    "more specifically", "on the contrary", "by contrast",
]

# Complex words: 3+ syllables (used in Fog Index)
_PASSIVE_RE = re.compile(
    r"\b(is|are|was|were|be|been|being)\s+"
    r"(\w+ed|shown|demonstrated|proposed|observed|found|given|known|"
    r"used|applied|performed|conducted|analyzed|measured|evaluated|"
    r"obtained|achieved|determined|compared|calculated|estimated)\b",
    re.IGNORECASE,
)

_NOMINALIZATION_SUFFIXES = ("tion", "sion", "ness", "ment", "ity", "ism",
                             "ance", "ence")


class ClarityScorer:
    """
    Analyze a document for per-sentence readability, repetition patterns,
    and paragraph-level coherence.

    Usage::

        scorer = ClarityScorer()
        report = scorer.analyze("Full document text here...")
        print(f"Clarity score: {report.overall_clarity_score}/100")
        for s in report.sentences:
            print(f"  Sentence {s.sentence_index}: FK={s.fk_grade:.1f}, "
                  f"complexity={s.complexity_score:.2f}")
    """

    def __init__(self, use_spacy: bool = True,
                 complexity_threshold: float = 0.7):
        self._nlp = None
        self._complexity_threshold = complexity_threshold
        if use_spacy:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm",
                                       disable=["ner", "lemmatizer"])
            except (ImportError, OSError):
                logger.debug("spaCy unavailable; clarity scoring uses regex.")

    def analyze(self, text: str) -> ClarityReport:
        """Run full clarity analysis on document text."""
        if not text or not text.strip():
            return ClarityReport()

        paragraphs = self._split_paragraphs(text)
        all_sentences: list[SentenceClarity] = []
        para_coherence: list[ParagraphCoherence] = []
        repetitions: list[RepetitionIssue] = []

        sent_idx = 0
        for p_idx, (para_text, para_start) in enumerate(paragraphs):
            para_sents: list[SentenceClarity] = []
            sents = self._split_sentences(para_text)

            for sent_text, rel_offset in sents:
                abs_offset = para_start + rel_offset
                sc = self._score_sentence(
                    sent_text, abs_offset, sent_idx, p_idx)
                para_sents.append(sc)
                all_sentences.append(sc)
                sent_idx += 1

            # Check for repetitions within paragraph
            repetitions.extend(self._find_repetitions(para_sents))

            # Paragraph coherence
            if para_sents:
                coh = self._score_paragraph_coherence(
                    para_sents, para_text, p_idx, para_start)
                para_coherence.append(coh)

        # Document-level metrics
        report = ClarityReport(
            sentences=all_sentences,
            repetitions=repetitions,
            paragraph_coherence=para_coherence,
            total_sentences=len(all_sentences),
            total_paragraphs=len(paragraphs),
        )

        if all_sentences:
            report.avg_fk_grade = round(
                sum(s.fk_grade for s in all_sentences) / len(all_sentences), 1)
            report.avg_fog_index = round(
                sum(s.fog_index for s in all_sentences) / len(all_sentences), 1)
            report.avg_sentence_words = round(
                sum(s.word_count for s in all_sentences) / len(all_sentences), 1)
            report.complex_sentence_count = sum(
                1 for s in all_sentences
                if s.complexity_score > self._complexity_threshold)

        report.overall_clarity_score = self._overall_score(report)
        return report

    # ------------------------------------------------------------------
    # Sentence-level scoring
    # ------------------------------------------------------------------

    def _score_sentence(self, text: str, start_offset: int,
                        sent_idx: int, para_idx: int) -> SentenceClarity:
        """Compute clarity metrics for a single sentence."""
        words = text.split()
        word_count = len(words)
        syllable_count = sum(self._count_syllables(w) for w in words)

        fk_grade = self._flesch_kincaid_grade(word_count, syllable_count, 1)
        fog_index = self._fog_index(text, word_count, 1)
        has_passive = bool(_PASSIVE_RE.search(text))
        nom_count = sum(1 for w in words
                        if w.lower().endswith(_NOMINALIZATION_SUFFIXES)
                        and len(w) > 6)

        complexity = self._complexity_score(
            word_count, syllable_count, fk_grade, has_passive, nom_count)

        return SentenceClarity(
            text=text,
            start_offset=start_offset,
            end_offset=start_offset + len(text),
            word_count=word_count,
            syllable_count=syllable_count,
            fk_grade=round(fk_grade, 1),
            fog_index=round(fog_index, 1),
            complexity_score=round(complexity, 3),
            has_passive_voice=has_passive,
            nominalization_count=nom_count,
            sentence_index=sent_idx,
            paragraph_index=para_idx,
        )

    def _complexity_score(self, word_count: int, syllable_count: int,
                          fk_grade: float, has_passive: bool,
                          nom_count: int) -> float:
        """
        Compute a 0–1 complexity score from multiple features.

        Higher = more complex / harder to read.
        """
        # Normalize each signal to 0–1 range
        len_signal = min(word_count / 60.0, 1.0)        # 60+ words = max
        syl_per_word = (syllable_count / max(word_count, 1))
        syl_signal = min((syl_per_word - 1.0) / 1.5, 1.0)  # avg 2.5+ syl = max
        fk_signal = min(max(fk_grade - 8, 0) / 12.0, 1.0)  # FK 20+ = max
        passive_signal = 0.3 if has_passive else 0.0
        nom_signal = min(nom_count / 4.0, 1.0)            # 4+ nominalizations = max

        # Weighted combination
        score = (
            0.30 * len_signal +
            0.20 * syl_signal +
            0.25 * fk_signal +
            0.10 * passive_signal +
            0.15 * nom_signal
        )
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Repetition detection
    # ------------------------------------------------------------------

    def _find_repetitions(self, sentences: list[SentenceClarity]
                          ) -> list[RepetitionIssue]:
        """Detect repetitive patterns between consecutive sentences."""
        results = []
        for i in range(len(sentences) - 1):
            a = sentences[i]
            b = sentences[i + 1]

            # 1. Vocabulary repetition: high word overlap
            words_a = set(re.findall(r"\b[a-z]{3,}\b", a.text.lower()))
            words_b = set(re.findall(r"\b[a-z]{3,}\b", b.text.lower()))
            if words_a and words_b:
                overlap = words_a & words_b
                # Exclude very common function words
                overlap -= {"the", "and", "that", "this", "with", "from",
                            "have", "been", "were", "are", "was", "for",
                            "not", "but", "can", "has"}
                ratio = len(overlap) / min(len(words_a), len(words_b))
                if ratio > 0.6 and len(overlap) >= 4:
                    shared = ", ".join(sorted(overlap)[:6])
                    results.append(RepetitionIssue(
                        sentence_a_index=a.sentence_index,
                        sentence_b_index=b.sentence_index,
                        sentence_a_text=a.text[:100],
                        sentence_b_text=b.text[:100],
                        repetition_type="vocabulary",
                        shared_content=shared,
                        severity=min(ratio, 1.0),
                        start_offset=a.start_offset,
                        end_offset=b.end_offset,
                    ))

            # 2. Opening repetition: same first 3 words
            first_a = " ".join(a.text.split()[:3]).lower()
            first_b = " ".join(b.text.split()[:3]).lower()
            if first_a == first_b and len(first_a) > 5:
                results.append(RepetitionIssue(
                    sentence_a_index=a.sentence_index,
                    sentence_b_index=b.sentence_index,
                    sentence_a_text=a.text[:100],
                    sentence_b_text=b.text[:100],
                    repetition_type="opening",
                    shared_content=first_a,
                    severity=0.6,
                    start_offset=a.start_offset,
                    end_offset=b.end_offset,
                ))

        return results

    # ------------------------------------------------------------------
    # Paragraph coherence
    # ------------------------------------------------------------------

    def _score_paragraph_coherence(self, sentences: list[SentenceClarity],
                                    para_text: str, para_idx: int,
                                    para_start: int) -> ParagraphCoherence:
        """Score how well sentences within a paragraph cohere."""
        if len(sentences) <= 1:
            return ParagraphCoherence(
                paragraph_index=para_idx,
                text_preview=para_text[:120],
                sentence_count=len(sentences),
                topic_consistency=1.0,
                transition_score=1.0,
                overall_coherence=1.0,
                start_offset=para_start,
                end_offset=para_start + len(para_text),
            )

        # Topic consistency: vocabulary overlap between adjacent sentences
        overlaps = []
        for i in range(len(sentences) - 1):
            words_a = set(re.findall(r"\b[a-z]{4,}\b", sentences[i].text.lower()))
            words_b = set(re.findall(r"\b[a-z]{4,}\b", sentences[i+1].text.lower()))
            if words_a and words_b:
                overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
                overlaps.append(overlap)
        topic_consistency = sum(overlaps) / len(overlaps) if overlaps else 0.5

        # Transition score: presence of discourse connectors
        para_lower = para_text.lower()
        transition_count = 0
        for word in _TRANSITION_WORDS:
            transition_count += len(re.findall(r"\b" + re.escape(word) + r"\b",
                                               para_lower))
        for phrase in _TRANSITION_PHRASES:
            transition_count += len(re.findall(re.escape(phrase), para_lower))

        expected_transitions = max(len(sentences) - 1, 1)
        transition_score = min(transition_count / expected_transitions, 1.0)

        # Overall coherence: weighted average
        overall = 0.6 * topic_consistency + 0.4 * transition_score

        return ParagraphCoherence(
            paragraph_index=para_idx,
            text_preview=para_text[:120],
            sentence_count=len(sentences),
            topic_consistency=round(topic_consistency, 3),
            transition_score=round(transition_score, 3),
            overall_coherence=round(overall, 3),
            start_offset=para_start,
            end_offset=para_start + len(para_text),
        )

    # ------------------------------------------------------------------
    # Readability formulas
    # ------------------------------------------------------------------

    @staticmethod
    def _flesch_kincaid_grade(words: int, syllables: int,
                               sentences: int) -> float:
        """Flesch-Kincaid Grade Level formula."""
        if words == 0 or sentences == 0:
            return 0.0
        return (0.39 * (words / sentences) +
                11.8 * (syllables / words) - 15.59)

    @staticmethod
    def _fog_index(text: str, word_count: int, sentence_count: int) -> float:
        """Gunning Fog Index."""
        if word_count == 0 or sentence_count == 0:
            return 0.0
        words = text.split()
        complex_words = sum(1 for w in words
                            if ClarityScorer._count_syllables(w) >= 3)
        return 0.4 * ((word_count / sentence_count) +
                       100 * (complex_words / word_count))

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Estimate syllable count for an English word."""
        word = word.lower().rstrip(".,;:!?\"')")
        if len(word) <= 3:
            return 1
        # Remove trailing silent 'e'
        if word.endswith("e") and not word.endswith("le"):
            word = word[:-1]
        # Count vowel groups
        count = len(re.findall(r"[aeiouy]+", word))
        return max(count, 1)

    # ------------------------------------------------------------------
    # Overall clarity score
    # ------------------------------------------------------------------

    def _overall_score(self, report: ClarityReport) -> float:
        """
        Compute a 0–100 overall clarity score.

        100 = perfectly clear, 0 = extremely hard to read.
        """
        if not report.sentences:
            return 100.0

        # Component scores (each 0–1, higher = better)
        n = len(report.sentences)

        # 1. Average complexity (inverted: lower complexity = higher score)
        avg_complexity = sum(s.complexity_score for s in report.sentences) / n
        complexity_component = 1.0 - avg_complexity

        # 2. FK Grade appropriateness (target: 12–16 for academic writing)
        fk = report.avg_fk_grade
        if 12 <= fk <= 16:
            fk_component = 1.0
        elif fk < 12:
            fk_component = max(0.5, fk / 12.0)
        else:
            fk_component = max(0.3, 1.0 - (fk - 16) / 10.0)

        # 3. Repetition penalty
        rep_penalty = min(len(report.repetitions) * 0.05, 0.3)
        repetition_component = 1.0 - rep_penalty

        # 4. Paragraph coherence
        if report.paragraph_coherence:
            coherence_component = sum(
                p.overall_coherence for p in report.paragraph_coherence
            ) / len(report.paragraph_coherence)
        else:
            coherence_component = 0.5

        # 5. Sentence length variety (moderate CV is good)
        lengths = [s.word_count for s in report.sentences]
        if len(lengths) > 1:
            mean_len = sum(lengths) / len(lengths)
            if mean_len > 0:
                std_len = (sum((l - mean_len)**2 for l in lengths)
                           / len(lengths)) ** 0.5
                cv = std_len / mean_len
                # Ideal CV is around 0.4–0.6
                if 0.3 <= cv <= 0.7:
                    variety_component = 1.0
                elif cv < 0.3:
                    variety_component = 0.7  # Too uniform
                else:
                    variety_component = max(0.5, 1.0 - (cv - 0.7) * 0.5)
            else:
                variety_component = 0.5
        else:
            variety_component = 0.5

        # Weighted combination
        score = (
            0.30 * complexity_component +
            0.20 * fk_component +
            0.15 * repetition_component +
            0.20 * coherence_component +
            0.15 * variety_component
        )

        return round(score * 100, 1)

    # ------------------------------------------------------------------
    # Text splitting helpers
    # ------------------------------------------------------------------

    def _split_paragraphs(self, text: str) -> list[tuple[str, int]]:
        """Split text into (paragraph_text, start_offset) pairs."""
        results = []
        pos = 0
        for para in re.split(r"\n\s*\n", text):
            para_stripped = para.strip()
            if para_stripped and len(para_stripped) > 20:
                idx = text.find(para_stripped[:30], pos)
                if idx >= 0:
                    results.append((para_stripped, idx))
                    pos = idx + len(para_stripped)
        return results

    def _split_sentences(self, text: str) -> list[tuple[str, int]]:
        """Split paragraph into (sentence_text, relative_offset) pairs."""
        if self._nlp:
            try:
                doc = self._nlp(text)
                return [(s.text.strip(), s.start_char)
                        for s in doc.sents if len(s.text.strip()) > 10]
            except Exception:
                pass
        # Regex fallback
        protected = re.sub(
            r"\b(e\.g|i\.e|et al|Fig|Tab|Eq|cf|vs|Dr|Mr|Mrs|Prof|al|approx)\.",
            lambda m: m.group(0).replace(".", "<DOT>"), text)
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)
        results = []
        pos = 0
        for part in parts:
            restored = part.replace("<DOT>", ".")
            s = restored.strip()
            if len(s) > 10:
                idx = text.find(s[:30], pos)
                if idx >= 0:
                    results.append((s, idx))
                    pos = idx + len(s)
        return results
