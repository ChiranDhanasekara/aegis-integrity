"""
AEGIS Writing Assistant Engine -- v4.0 Novel Feature.

Rule-based academic writing improvement that produces actionable,
reviewable suggestions. This module does NOT use any external LLM API;
every transformation is a deterministic pattern-match or spaCy-based
heuristic that runs fully offline.

Components:
    suggestion.py      -- WritingSuggestion data model and management
    rewriter.py        -- Sentence-level rewrite engine (passive→active,
                          wordiness, nominalizations, hedging, etc.)
    clarity_scorer.py  -- Per-sentence readability, repetition detection,
                          paragraph coherence scoring

The Writing Assistant focuses on clarity, grammar, and academic style.
It does NOT rephrase text to evade plagiarism detection -- suggestions
always preserve the author's meaning while improving expression quality.
"""

from aegis.writing.suggestion import WritingSuggestion, SuggestionSet
from aegis.writing.rewriter import AcademicRewriter
from aegis.writing.clarity_scorer import ClarityScorer

__all__ = [
    "WritingSuggestion",
    "SuggestionSet",
    "AcademicRewriter",
    "ClarityScorer",
]
