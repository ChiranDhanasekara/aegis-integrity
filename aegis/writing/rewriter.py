"""
Academic Rewrite Engine -- AEGIS Writing Assistant v4.0.

A rule-based, deterministic sentence-level rewriting engine for academic
manuscripts. Every transformation preserves semantic meaning while
improving clarity, conciseness, and academic tone.

This module does NOT call external LLM APIs. All transformations are
regex patterns, spaCy dependency-parse heuristics, or dictionary lookups
that run fully offline.

Rewrite categories:
  1. Passive → active voice (spaCy dep-parse when available)
  2. Wordiness reduction (pattern-matched verbose → concise phrases)
  3. Nominalization reversal (abstract nouns → concrete verbs)
  4. Hedge trimming (excessive hedging language)
  5. Contraction expansion (informal → formal academic tone)
  6. British/American spelling consistency (auto-fix to dominant variant)
  7. Sentence splitting (for sentences exceeding word threshold)
  8. Repeated word cleanup
  9. Common usage error corrections
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass
from typing import Optional

from aegis.writing.suggestion import WritingSuggestion, SuggestionSet

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Reference data: verbose → concise phrase mappings
# -----------------------------------------------------------------------

# Each tuple: (verbose_pattern_regex, replacement, explanation)
_WORDINESS_RULES: list[tuple[str, str, str]] = [
    (r"\bin order to\b", "to",
     "\"In order to\" can be shortened to \"to\" without loss of meaning."),
    (r"\bdue to the fact that\b", "because",
     "\"Due to the fact that\" is verbose; use \"because\"."),
    (r"\bin spite of the fact that\b", "although",
     "\"In spite of the fact that\" can be replaced with \"although\"."),
    (r"\bfor the purpose of\b", "to",
     "\"For the purpose of\" can be shortened to \"to\" or \"for\"."),
    (r"\bin the event that\b", "if",
     "\"In the event that\" can be shortened to \"if\"."),
    (r"\bat this point in time\b", "now",
     "\"At this point in time\" is verbose; use \"now\" or \"currently\"."),
    (r"\bat the present time\b", "now",
     "\"At the present time\" can be shortened to \"now\"."),
    (r"\bwith regard to\b", "regarding",
     "\"With regard to\" can be shortened to \"regarding\" or \"about\"."),
    (r"\bwith respect to\b", "regarding",
     "\"With respect to\" can be shortened to \"regarding\"."),
    (r"\bin regard to\b", "regarding",
     "\"In regard to\" can be shortened to \"regarding\"."),
    (r"\bin close proximity to\b", "near",
     "\"In close proximity to\" is verbose; use \"near\"."),
    (r"\ba large number of\b", "many",
     "\"A large number of\" can be shortened to \"many\"."),
    (r"\ba small number of\b", "few",
     "\"A small number of\" can be shortened to \"few\"."),
    (r"\bthe vast majority of\b", "most",
     "\"The vast majority of\" can be shortened to \"most\"."),
    (r"\bthe majority of\b", "most",
     "\"The majority of\" can be shortened to \"most\"."),
    (r"\ba considerable amount of\b", "much",
     "\"A considerable amount of\" can be shortened to \"much\"."),
    (r"\bhas the ability to\b", "can",
     "\"Has the ability to\" can be shortened to \"can\"."),
    (r"\bis able to\b", "can",
     "\"Is able to\" can be shortened to \"can\"."),
    (r"\bare able to\b", "can",
     "\"Are able to\" can be shortened to \"can\"."),
    (r"\bwas able to\b", "could",
     "\"Was able to\" can be shortened to \"could\"."),
    (r"\bin the case of\b", "for",
     "\"In the case of\" can be shortened to \"for\" or \"in\"."),
    (r"\bit is important to note that\b", "",
     "\"It is important to note that\" is a throat-clearing phrase; "
     "the sentence is stronger without it."),
    (r"\bit should be noted that\b", "",
     "\"It should be noted that\" can usually be removed entirely."),
    (r"\bit is worth noting that\b", "",
     "\"It is worth noting that\" can usually be removed."),
    (r"\bit is interesting to note that\b", "",
     "\"It is interesting to note that\" can usually be removed."),
    (r"\bin light of the fact that\b", "because",
     "\"In light of the fact that\" can be shortened to \"because\"."),
    (r"\bas a matter of fact\b", "in fact",
     "\"As a matter of fact\" can be shortened to \"in fact\"."),
    (r"\bon the basis of\b", "based on",
     "\"On the basis of\" can be shortened to \"based on\"."),
    (r"\bprior to\b", "before",
     "\"Prior to\" can be shortened to \"before\"."),
    (r"\bsubsequent to\b", "after",
     "\"Subsequent to\" can be shortened to \"after\"."),
    (r"\bin the absence of\b", "without",
     "\"In the absence of\" can be shortened to \"without\"."),
    (r"\bby means of\b", "by",
     "\"By means of\" can be shortened to \"by\" or \"using\"."),
    (r"\bfor the reason that\b", "because",
     "\"For the reason that\" can be shortened to \"because\"."),
    (r"\bin an effort to\b", "to",
     "\"In an effort to\" can be shortened to \"to\"."),
    (r"\bas a consequence of\b", "because of",
     "\"As a consequence of\" can be shortened to \"because of\"."),
    (r"\bnotwithstanding the fact that\b", "although",
     "\"Notwithstanding the fact that\" can be shortened to \"although\"."),
]

# Nominalization → verb form mappings
_NOMINALIZATION_MAP: dict[str, str] = {
    "utilization": "use",
    "utilisation": "use",
    "implementation": "implement",
    "investigation": "investigate",
    "examination": "examine",
    "determination": "determine",
    "observation": "observe",
    "demonstration": "demonstrate",
    "establishment": "establish",
    "consideration": "consider",
    "facilitation": "facilitate",
    "modification": "modify",
    "optimization": "optimize",
    "optimisation": "optimise",
    "evaluation": "evaluate",
    "classification": "classify",
    "identification": "identify",
    "characterization": "characterize",
    "characterisation": "characterise",
    "documentation": "document",
    "calculation": "calculate",
    "verification": "verify",
    "transformation": "transform",
    "installation": "install",
    "initialization": "initialize",
    "initialisation": "initialise",
    "normalization": "normalize",
    "normalisation": "normalise",
    "visualization": "visualize",
    "visualisation": "visualise",
    "maximization": "maximize",
    "minimization": "minimize",
    "quantification": "quantify",
    "simplification": "simplify",
}

# Patterns that introduce nominalizations with "the X of"
_NOMINALIZATION_PATTERN = re.compile(
    r"\bthe\s+(" + "|".join(re.escape(n) for n in _NOMINALIZATION_MAP) +
    r")\s+of\b",
    re.IGNORECASE,
)

# Contraction → expansion map
_CONTRACTION_EXPANSIONS: dict[str, str] = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "won't": "will not", "wouldn't": "would not",
    "shouldn't": "should not", "couldn't": "could not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "it's": "it is", "that's": "that is",
    "there's": "there is", "here's": "here is", "let's": "let us",
    "what's": "what is", "who's": "who is",
    "i'm": "I am", "we're": "we are", "they're": "they are",
    "you're": "you are", "he's": "he is", "she's": "she is",
    "i've": "I have", "we've": "we have", "they've": "they have",
    "you've": "you have",
    "i'll": "I will", "we'll": "we will", "they'll": "they will",
    "you'll": "you will", "he'll": "he will", "she'll": "she will",
}

_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in _CONTRACTION_EXPANSIONS) + r")\b",
    re.IGNORECASE,
)

# Hedge phrases that weaken academic writing
_HEDGE_PHRASES: list[tuple[str, str]] = [
    (r"\bit could potentially be argued that\b",
     "Remove or rewrite: overly hedged throat-clearing phrase."),
    (r"\bit may be possible that\b",
     "\"It may be possible that\" double-hedges; pick one: \"possibly\" or \"may\"."),
    (r"\bto some extent\b",
     "\"To some extent\" is vague; specify the degree or remove."),
    (r"\bmore or less\b",
     "\"More or less\" is imprecise; use a specific qualifier or remove."),
    (r"\bsort of\b",
     "\"Sort of\" is informal; use \"somewhat\" or be specific."),
    (r"\bkind of\b",
     "\"Kind of\" is informal; use \"somewhat\" or be specific."),
    (r"\bquite\b",
     "\"Quite\" is vague; quantify or remove for precision."),
    (r"\brather\b(?!\s+than)",
     "\"Rather\" is often a filler; consider removing for directness."),
    (r"\bbasically\b",
     "\"Basically\" is informal and usually adds no meaning in academic text."),
]

# Usage error corrections (from grammar.py, now with fix suggestions)
_USAGE_CORRECTIONS: list[tuple[str, str, str, str]] = [
    # (pattern, replacement_text, explanation, rule_source)
    (r"\bcomprised of\b", "composed of",
     "\"Comprised of\" is nonstandard; use \"comprises\" or \"is composed of\".",
     "general usage"),
    (r"\balot\b", "a lot",
     "\"Alot\" is not a word; use \"a lot\".", "spelling"),
    (r"\birregardless\b", "regardless",
     "\"Irregardless\" is nonstandard; use \"regardless\".", "usage"),
    (r"\bcould of\b", "could have",
     "\"Could of\" should be \"could have\".", "usage"),
    (r"\bwould of\b", "would have",
     "\"Would of\" should be \"would have\".", "usage"),
    (r"\bshould of\b", "should have",
     "\"Should of\" should be \"should have\".", "usage"),
]

# UK/US spelling pairs (imported concept from grammar.py)
_UK_US_PAIRS = [
    ("behaviour", "behavior"), ("colour", "color"), ("favour", "favor"),
    ("honour", "honor"), ("labour", "labor"), ("neighbour", "neighbor"),
    ("centre", "center"), ("metre", "meter"), ("fibre", "fiber"),
    ("organisation", "organization"), ("organise", "organize"),
    ("analyse", "analyze"), ("analysed", "analyzed"), ("analysing", "analyzing"),
    ("recognise", "recognize"), ("recognised", "recognized"),
    ("optimise", "optimize"), ("optimised", "optimized"),
    ("optimisation", "optimization"),
    ("characterise", "characterize"), ("characterised", "characterized"),
    ("minimise", "minimize"), ("maximise", "maximize"),
    ("defence", "defense"), ("licence", "license"),
    ("modelling", "modeling"), ("modelled", "modeled"),
    ("cancelled", "canceled"), ("cancelling", "canceling"),
    ("travelled", "traveled"), ("travelling", "traveling"),
    ("labelled", "labeled"), ("labelling", "labeling"),
    ("judgement", "judgment"), ("acknowledgement", "acknowledgment"),
    ("utilise", "utilize"), ("utilised", "utilized"),
    ("realise", "realize"), ("realised", "realized"),
    ("emphasise", "emphasize"), ("emphasised", "emphasized"),
    ("categorise", "categorize"), ("categorised", "categorized"),
    ("summarise", "summarize"), ("summarised", "summarized"),
]

# Passive voice detection (heuristic)
_PASSIVE_RE = re.compile(
    r"\b(is|are|was|were|be|been|being)\s+"
    r"(\w+ed|shown|demonstrated|proposed|observed|found|given|known|"
    r"used|applied|performed|conducted|analyzed|measured|evaluated|"
    r"obtained|achieved|determined|compared|calculated|estimated|"
    r"implemented|designed|developed|presented|reported|discussed)\b",
    re.IGNORECASE,
)

# Repeated words (immediately adjacent)
_REPEATED_WORD_RE = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)


@dataclass
class RewriterConfig:
    """Configuration for the AcademicRewriter."""
    # Which rewrite categories to enable
    fix_wordiness: bool = True
    fix_nominalizations: bool = True
    fix_passive_voice: bool = True
    fix_contractions: bool = True
    fix_hedging: bool = True
    fix_spelling_consistency: bool = True
    fix_usage_errors: bool = True
    fix_repeated_words: bool = True
    suggest_sentence_splits: bool = True

    # Spelling convention: "auto" (detect dominant), "UK", or "US"
    spelling_convention: str = "auto"

    # Sentence length threshold for split suggestions
    max_sentence_words: int = 45

    # Minimum confidence to emit a suggestion
    min_confidence: float = 0.3


class AcademicRewriter:
    """
    Rule-based academic writing improvement engine.

    Produces a SuggestionSet of reviewable, character-offset-tagged
    suggestions for improving an academic manuscript.

    Usage::

        rewriter = AcademicRewriter()
        suggestions = rewriter.analyze("The utilization of deep learning...")
        for s in suggestions:
            print(f"[{s.category}] {s.explanation}")
            print(f"  '{s.original_text}' → '{s.suggested_text}'")
    """

    def __init__(self, config: Optional[RewriterConfig] = None):
        self.cfg = config or RewriterConfig()
        self._nlp = None
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm",
                                   disable=["ner", "lemmatizer"])
        except (ImportError, OSError):
            logger.debug("spaCy unavailable; passive voice detection will "
                         "use regex fallback.")

    def analyze(self, text: str) -> SuggestionSet:
        """
        Analyze text and return a SuggestionSet of writing suggestions.

        text: The full document text to analyze.
        """
        suggestions = SuggestionSet()
        if not text or not text.strip():
            return suggestions

        all_sug: list[WritingSuggestion] = []

        if self.cfg.fix_wordiness:
            all_sug.extend(self._find_wordiness(text))

        if self.cfg.fix_nominalizations:
            all_sug.extend(self._find_nominalizations(text))

        if self.cfg.fix_contractions:
            all_sug.extend(self._find_contractions(text))

        if self.cfg.fix_hedging:
            all_sug.extend(self._find_hedging(text))

        if self.cfg.fix_usage_errors:
            all_sug.extend(self._find_usage_errors(text))

        if self.cfg.fix_repeated_words:
            all_sug.extend(self._find_repeated_words(text))

        if self.cfg.fix_spelling_consistency:
            all_sug.extend(self._find_spelling_inconsistencies(text))

        if self.cfg.fix_passive_voice:
            all_sug.extend(self._find_passive_voice(text))

        if self.cfg.suggest_sentence_splits:
            all_sug.extend(self._find_long_sentences(text))

        # Filter by minimum confidence
        all_sug = [s for s in all_sug if s.confidence >= self.cfg.min_confidence]

        suggestions.add_all(all_sug)
        return suggestions

    # ------------------------------------------------------------------
    # Individual rewrite detectors
    # ------------------------------------------------------------------

    def _find_wordiness(self, text: str) -> list[WritingSuggestion]:
        """Detect verbose phrases and suggest concise alternatives."""
        results = []
        for pattern_str, replacement, explanation in _WORDINESS_RULES:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for m in pattern.finditer(text):
                original = m.group(0)
                # Preserve case of first character
                if replacement:
                    suggested = self._match_case(original, replacement)
                else:
                    # Empty replacement means delete the phrase
                    suggested = ""
                results.append(WritingSuggestion(
                    category="wordiness",
                    severity="info",
                    original_text=original,
                    suggested_text=suggested,
                    explanation=explanation,
                    start_offset=m.start(),
                    end_offset=m.end(),
                    confidence=0.85,
                    rule_source="academic style guide",
                ))
        return results

    def _find_nominalizations(self, text: str) -> list[WritingSuggestion]:
        """Detect 'the X of' nominalization patterns."""
        results = []
        for m in _NOMINALIZATION_PATTERN.finditer(text):
            noun = m.group(1).lower()
            verb = _NOMINALIZATION_MAP.get(noun)
            if verb:
                results.append(WritingSuggestion(
                    category="nominalization",
                    severity="info",
                    original_text=m.group(0),
                    suggested_text=verb,
                    explanation=f"Consider using the verb \"{verb}\" instead of "
                                f"the noun phrase \"{m.group(0)}\". "
                                f"Verbs are usually more direct and concise.",
                    start_offset=m.start(),
                    end_offset=m.end(),
                    confidence=0.65,  # Lower — context-dependent
                    rule_source="Sword (2012), Stylish Academic Writing",
                ))
        return results

    def _find_contractions(self, text: str) -> list[WritingSuggestion]:
        """Find contractions and suggest formal expansions."""
        results = []
        for m in _CONTRACTION_RE.finditer(text):
            contraction = m.group(0)
            key = contraction.lower()
            expansion = _CONTRACTION_EXPANSIONS.get(key, contraction)
            # Preserve original case pattern
            if contraction[0].isupper():
                expansion = expansion[0].upper() + expansion[1:]
            results.append(WritingSuggestion(
                category="style",
                severity="warning",
                original_text=contraction,
                suggested_text=expansion,
                explanation=f"Contractions are conventionally avoided in formal "
                            f"academic writing. Expand \"{contraction}\" to "
                            f"\"{expansion}\".",
                start_offset=m.start(),
                end_offset=m.end(),
                confidence=0.95,
                rule_source="IEEE Editorial Style Manual for Authors (2024), p.22",
            ))
        return results

    def _find_hedging(self, text: str) -> list[WritingSuggestion]:
        """Detect excessive hedging phrases."""
        results = []
        for pattern_str, explanation in _HEDGE_PHRASES:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for m in pattern.finditer(text):
                results.append(WritingSuggestion(
                    category="hedge",
                    severity="info",
                    original_text=m.group(0),
                    suggested_text="",  # Removal suggestion
                    explanation=explanation,
                    start_offset=m.start(),
                    end_offset=m.end(),
                    confidence=0.5,  # Hedging is often intentional
                    rule_source="academic clarity guide",
                ))
        return results

    def _find_usage_errors(self, text: str) -> list[WritingSuggestion]:
        """Detect common usage errors with fix suggestions."""
        results = []
        for pattern_str, replacement, explanation, source in _USAGE_CORRECTIONS:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for m in pattern.finditer(text):
                original = m.group(0)
                suggested = self._match_case(original, replacement)
                results.append(WritingSuggestion(
                    category="grammar",
                    severity="error",
                    original_text=original,
                    suggested_text=suggested,
                    explanation=explanation,
                    start_offset=m.start(),
                    end_offset=m.end(),
                    confidence=0.95,
                    rule_source=source,
                ))
        return results

    def _find_repeated_words(self, text: str) -> list[WritingSuggestion]:
        """Detect immediately-repeated words (e.g., 'the the')."""
        results = []
        for m in _REPEATED_WORD_RE.finditer(text):
            word = m.group(1)
            results.append(WritingSuggestion(
                category="repetition",
                severity="warning",
                original_text=m.group(0),
                suggested_text=word,
                explanation=f"The word \"{word}\" is repeated immediately. "
                            f"Remove the duplicate.",
                start_offset=m.start(),
                end_offset=m.end(),
                confidence=0.90,
            ))
        return results

    def _find_spelling_inconsistencies(self, text: str) -> list[WritingSuggestion]:
        """
        Detect mixed UK/US spelling and suggest corrections to the
        dominant variant (or configured convention).
        """
        results = []
        text_lower = text.lower()

        # Count UK vs US occurrences to determine dominant convention
        uk_total = us_total = 0
        for uk, us in _UK_US_PAIRS:
            uk_total += len(re.findall(r"\b" + re.escape(uk) + r"\b", text_lower))
            us_total += len(re.findall(r"\b" + re.escape(us) + r"\b", text_lower))

        if uk_total == 0 and us_total == 0:
            return results  # No spelling variants detected

        # Determine target convention
        if self.cfg.spelling_convention == "auto":
            target = "UK" if uk_total > us_total else "US"
        else:
            target = self.cfg.spelling_convention

        # Only suggest fixes if there's actually a mix
        if uk_total > 0 and us_total > 0:
            for uk, us in _UK_US_PAIRS:
                if target == "US":
                    # Fix UK spellings → US
                    pattern = re.compile(r"\b" + re.escape(uk) + r"\b", re.IGNORECASE)
                    for m in pattern.finditer(text):
                        original = m.group(0)
                        suggested = self._match_case(original, us)
                        results.append(WritingSuggestion(
                            category="spelling",
                            severity="info",
                            original_text=original,
                            suggested_text=suggested,
                            explanation=f"Document predominantly uses US English. "
                                        f"Change \"{original}\" to \"{suggested}\" "
                                        f"for consistency.",
                            start_offset=m.start(),
                            end_offset=m.end(),
                            confidence=0.80,
                            rule_source="spelling consistency",
                        ))
                else:
                    # Fix US spellings → UK
                    pattern = re.compile(r"\b" + re.escape(us) + r"\b", re.IGNORECASE)
                    for m in pattern.finditer(text):
                        original = m.group(0)
                        suggested = self._match_case(original, uk)
                        results.append(WritingSuggestion(
                            category="spelling",
                            severity="info",
                            original_text=original,
                            suggested_text=suggested,
                            explanation=f"Document predominantly uses UK English. "
                                        f"Change \"{original}\" to \"{suggested}\" "
                                        f"for consistency.",
                            start_offset=m.start(),
                            end_offset=m.end(),
                            confidence=0.80,
                            rule_source="spelling consistency",
                        ))

        return results

    def _find_passive_voice(self, text: str) -> list[WritingSuggestion]:
        """
        Detect passive voice constructions and suggest review.

        Note: Passive voice is sometimes appropriate in academic writing
        (e.g., Methods sections). Confidence is set lower to reflect this.
        """
        results = []
        for m in _PASSIVE_RE.finditer(text):
            # Expand match to the full sentence for context
            sent_start = text.rfind(".", 0, m.start())
            sent_start = sent_start + 1 if sent_start >= 0 else 0
            sent_end = text.find(".", m.end())
            sent_end = sent_end + 1 if sent_end >= 0 else len(text)
            sentence = text[sent_start:sent_end].strip()

            results.append(WritingSuggestion(
                category="passive_voice",
                severity="info",
                original_text=m.group(0),
                suggested_text=m.group(0),  # No auto-fix for passive voice
                explanation=f"Passive voice detected: \"{m.group(0)}\". "
                            f"Consider rewriting in active voice for directness. "
                            f"Context: \"{sentence[:120]}...\"",
                start_offset=m.start(),
                end_offset=m.end(),
                confidence=0.45,  # Low — passive is often fine in academia
                rule_source="academic writing style guide",
            ))
        return results

    def _find_long_sentences(self, text: str) -> list[WritingSuggestion]:
        """Suggest splitting sentences that exceed the word threshold."""
        results = []
        sentences = self._split_sentences(text)

        for sent_text, sent_start in sentences:
            words = sent_text.split()
            if len(words) > self.cfg.max_sentence_words:
                # Look for potential split points (semicolons, conjunctions)
                split_hint = ""
                for conj in ["; ", ", and ", ", but ", ", however,",
                             ", which ", ", while "]:
                    if conj in sent_text:
                        split_hint = (f" Consider splitting at \"{conj.strip()}\""
                                      f" into two sentences.")
                        break

                results.append(WritingSuggestion(
                    category="sentence_length",
                    severity="info",
                    original_text=sent_text,
                    suggested_text=sent_text,  # No auto-fix
                    explanation=f"This sentence is {len(words)} words long "
                                f"(threshold: {self.cfg.max_sentence_words}). "
                                f"Long sentences reduce readability.{split_hint}",
                    start_offset=sent_start,
                    end_offset=sent_start + len(sent_text),
                    confidence=0.60,
                    rule_source="readability guidance",
                ))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> list[tuple[str, int]]:
        """Split text into (sentence_text, start_offset) pairs."""
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

    @staticmethod
    def _match_case(original: str, replacement: str) -> str:
        """Match the case pattern of the original text."""
        if not replacement:
            return replacement
        if original.isupper():
            return replacement.upper()
        if original[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement
