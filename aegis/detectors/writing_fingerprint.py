"""
Multi-Dimensional Stylometric Writing Fingerprinter -- AEGIS v4.0.

Extracts authorial style vectors and compares submissions against:
  1. An author's historical baseline profile (verifying authenticity / ghostwriting detection)
  2. Internal segment-to-segment consistency (detecting spliced AI or copied sections)

Metrics:
  - 50-dimensional function word frequency distribution (Burrows' Delta)
  - Sentence length statistics (mean, standard deviation, coefficient of variation)
  - Vocabulary richness: Type-Token Ratio (TTR), Hapax Legomena ratio, Yule's Characteristic K
  - Punctuation cadence: comma, semicolon, dash, and parenthesis frequencies per 100 words
  - Structural traits: passive voice ratio, nominalization density, syllables per word
"""

from __future__ import annotations
import re
import math
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional, Union


# Top 50 English function words for authorship attribution
TOP_FUNCTION_WORDS = [
    "the", "of", "and", "to", "a", "in", "that", "is", "was", "he",
    "for", "it", "with", "as", "his", "on", "be", "at", "by", "i",
    "this", "had", "not", "are", "but", "from", "or", "have", "an", "they",
    "which", "one", "you", "were", "her", "all", "she", "there", "would", "their",
    "we", "him", "been", "has", "when", "who", "will", "more", "no", "if"
]


@dataclass
class StyleVector:
    """Multi-dimensional stylistic representation of an author's text."""
    word_count: int
    sentence_count: int
    mean_sentence_length: float
    std_sentence_length: float
    type_token_ratio: float
    hapax_legomena_ratio: float
    yules_k: float
    function_word_frequencies: dict[str, float] = field(default_factory=dict)
    punctuation_frequencies: dict[str, float] = field(default_factory=dict)
    passive_voice_ratio: float = 0.0
    nominalization_density: float = 0.0
    avg_syllables_per_word: float = 0.0

    def to_dict(self) -> dict:
        return {
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "mean_sentence_length": round(self.mean_sentence_length, 2),
            "std_sentence_length": round(self.std_sentence_length, 2),
            "ttr": round(self.type_token_ratio, 3),
            "hapax_ratio": round(self.hapax_legomena_ratio, 3),
            "yules_k": round(self.yules_k, 2),
            "passive_voice_ratio": round(self.passive_voice_ratio, 3),
            "nominalization_density": round(self.nominalization_density, 3),
            "avg_syllables": round(self.avg_syllables_per_word, 2),
        }


@dataclass
class ComparisonResult:
    """Result of comparing two style vectors."""
    distance_score: float         # 0.0 (identical) to 1.0 (completely distinct)
    burrows_delta: float          # Standard Burrows' Delta distance
    cosine_distance: float
    is_consistent: bool           # True if style matches within normal variance
    divergence_areas: list[str] = field(default_factory=list)


class WritingFingerprinter:
    """
    Extracts stylistic fingerprints and compares text segments or author baselines.

    Usage::

        fingerprinter = WritingFingerprinter()
        vec_a = fingerprinter.extract_vector(prior_paper_text)
        vec_b = fingerprinter.extract_vector(submission_text)
        result = fingerprinter.compare_vectors(vec_a, vec_b)
        print(f"Distance: {result.distance_score:.2f}, Consistent: {result.is_consistent}")
    """

    def __init__(self, consistency_threshold: float = 0.35):
        self.consistency_threshold = consistency_threshold

    def extract_vector(self, text: str) -> StyleVector:
        """Extract multi-feature stylistic vector from text."""
        if not text or not text.strip():
            return StyleVector(
                word_count=0,
                sentence_count=0,
                mean_sentence_length=0.0,
                std_sentence_length=0.0,
                type_token_ratio=0.0,
                hapax_legomena_ratio=0.0,
                yules_k=0.0,
            )

        words = re.findall(r"\b[a-z']+\b", text.lower())
        word_count = len(words)
        if word_count == 0:
            return StyleVector(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Sentence parsing
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip().split()) >= 3]
        sentence_count = max(len(sentences), 1)

        sentence_lengths = [len(re.findall(r"\b[a-z']+\b", s.lower())) for s in sentences]
        mean_len = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
        
        variance = sum((l - mean_len) ** 2 for l in sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
        std_len = math.sqrt(variance)

        # Vocabulary richness metrics
        word_counts = Counter(words)
        unique_words = len(word_counts)
        ttr = unique_words / word_count
        
        hapax_count = sum(1 for w, c in word_counts.items() if c == 1)
        hapax_ratio = hapax_count / word_count

        # Yule's K
        # K = 10^4 * (sum(i^2 * V_i) - N) / N^2
        freq_spectrum = Counter(word_counts.values())
        sum_spectrum = sum((freq ** 2) * count for freq, count in freq_spectrum.items())
        yules_k = 10000 * (sum_spectrum - word_count) / (word_count ** 2) if word_count > 1 else 0.0

        # Function word frequencies (per 1000 words)
        fw_freqs = {}
        for fw in TOP_FUNCTION_WORDS:
            fw_freqs[fw] = (word_counts.get(fw, 0) / word_count) * 1000.0

        # Punctuation frequencies (per 100 words)
        punc_chars = {",": "comma", ";": "semicolon", ":": "colon", "-": "dash", "(": "paren"}
        punc_freqs = {}
        for char, name in punc_chars.items():
            punc_freqs[name] = (text.count(char) / word_count) * 100.0

        # Passive voice ratio
        passive_matches = len(re.findall(r"\b(?:is|are|was|were|be|been|being)\s+\w+ed\b", text.lower()))
        passive_ratio = passive_matches / sentence_count

        # Nominalization density (words ending in -tion, -ment, -ance, -ence per 100 words)
        nom_count = len(re.findall(r"\b\w{4,}(?:tion|tions|ment|ments|ance|ances|ence|ences)\b", text.lower()))
        nom_density = (nom_count / word_count) * 100.0

        # Syllables per word estimate
        total_syllables = sum(self._count_syllables(w) for w in words)
        avg_syllables = total_syllables / word_count

        return StyleVector(
            word_count=word_count,
            sentence_count=sentence_count,
            mean_sentence_length=mean_len,
            std_sentence_length=std_len,
            type_token_ratio=ttr,
            hapax_legomena_ratio=hapax_ratio,
            yules_k=yules_k,
            function_word_frequencies=fw_freqs,
            punctuation_frequencies=punc_freqs,
            passive_voice_ratio=passive_ratio,
            nominalization_density=nom_density,
            avg_syllables_per_word=avg_syllables,
        )

    def compare_vectors(self, vec_a: StyleVector, vec_b: StyleVector) -> ComparisonResult:
        """Compute statistical distance between two style vectors."""
        if vec_a.word_count == 0 or vec_b.word_count == 0:
            return ComparisonResult(
                distance_score=0.0,
                burrows_delta=0.0,
                cosine_distance=0.0,
                is_consistent=True,
            )

        # 1. Burrows' Delta on Function Words
        delta_sum = 0.0
        for fw in TOP_FUNCTION_WORDS:
            fa = vec_a.function_word_frequencies.get(fw, 0.0)
            fb = vec_b.function_word_frequencies.get(fw, 0.0)
            delta_sum += abs(fa - fb)
        burrows_delta = delta_sum / len(TOP_FUNCTION_WORDS)

        # 2. Cosine Distance on Function Word Vectors
        dot = sum(vec_a.function_word_frequencies.get(fw, 0.0) * vec_b.function_word_frequencies.get(fw, 0.0) for fw in TOP_FUNCTION_WORDS)
        mag_a = math.sqrt(sum(v ** 2 for v in vec_a.function_word_frequencies.values()))
        mag_b = math.sqrt(sum(v ** 2 for v in vec_b.function_word_frequencies.values()))
        cosine_sim = (dot / (mag_a * mag_b)) if (mag_a and mag_b) else 1.0
        cosine_dist = max(0.0, 1.0 - cosine_sim)

        # 3. Structural feature divergence checks
        divergences = []
        
        # Sentence length shift (>40% change)
        if vec_a.mean_sentence_length > 0:
            len_shift = abs(vec_a.mean_sentence_length - vec_b.mean_sentence_length) / vec_a.mean_sentence_length
            if len_shift > 0.45:
                divergences.append(f"Significant sentence length shift: {vec_a.mean_sentence_length:.1f} vs {vec_b.mean_sentence_length:.1f} words")

        # Vocabulary richness shift
        ttr_shift = abs(vec_a.type_token_ratio - vec_b.type_token_ratio)
        if ttr_shift > 0.15:
            divergences.append(f"Vocabulary richness discrepancy (TTR diff: {ttr_shift:.2f})")

        # Function word divergence
        if burrows_delta > 6.0:
            divergences.append(f"Function word distribution deviation (Delta: {burrows_delta:.2f})")

        # Composite distance normalized to 0.0-1.0
        norm_delta = min(burrows_delta / 12.0, 1.0)
        distance_score = round(0.50 * norm_delta + 0.30 * cosine_dist + 0.20 * min(len(divergences) * 0.3, 1.0), 3)

        is_consistent = distance_score < self.consistency_threshold

        return ComparisonResult(
            distance_score=distance_score,
            burrows_delta=round(burrows_delta, 3),
            cosine_distance=round(cosine_dist, 3),
            is_consistent=is_consistent,
            divergence_areas=divergences,
        )

    def analyze_segments(self, text: str, segment_size_words: int = 300) -> list[ComparisonResult]:
        """Split text into segments and compare adjacent blocks to detect stylistic shifts."""
        words = text.split()
        if len(words) < segment_size_words * 2:
            return []

        segments = []
        min_words = min(segment_size_words // 2, 50)
        for i in range(0, len(words), segment_size_words):
            chunk = " ".join(words[i:i + segment_size_words])
            if len(chunk.split()) >= min_words:
                segments.append(chunk)

        results = []
        for i in range(len(segments) - 1):
            va = self.extract_vector(segments[i])
            vb = self.extract_vector(segments[i + 1])
            res = self.compare_vectors(va, vb)
            results.append(res)

        return results

    def _count_syllables(self, word: str) -> int:
        """Heuristic syllable counting."""
        w = word.lower().strip()
        if len(w) <= 3:
            return 1
        w = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', w)
        w = re.sub(r'^y', '', w)
        syllables = len(re.findall(r'[aeiouy]{1,2}', w))
        return max(syllables, 1)
