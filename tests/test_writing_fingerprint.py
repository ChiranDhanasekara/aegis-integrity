"""
Unit tests for aegis.detectors.writing_fingerprint and sentence-level AI detection.
"""

import pytest
from aegis.detectors.writing_fingerprint import (
    StyleVector,
    ComparisonResult,
    WritingFingerprinter,
)
from aegis.detectors.ai_detector import AIContentDetector, SentenceAIScore


class TestWritingFingerprinter:

    def setup_method(self):
        self.fingerprinter = WritingFingerprinter(consistency_threshold=0.35)

    def test_extract_vector_properties(self):
        text = """
        Convolutional neural networks represent an effective architecture for computer vision tasks.
        In this study, we evaluate several model variants across multiple benchmark datasets.
        The empirical results demonstrate consistent performance improvements.
        """
        vec = self.fingerprinter.extract_vector(text)
        assert isinstance(vec, StyleVector)
        assert vec.word_count > 20
        assert vec.sentence_count >= 2
        assert vec.mean_sentence_length > 5
        assert 0.0 < vec.type_token_ratio <= 1.0
        assert "the" in vec.function_word_frequencies
        assert "and" in vec.function_word_frequencies

    def test_empty_text(self):
        vec = self.fingerprinter.extract_vector("")
        assert vec.word_count == 0
        res = self.fingerprinter.compare_vectors(vec, vec)
        assert res.is_consistent is True
        assert res.distance_score == 0.0

    def test_compare_same_text_returns_zero_distance(self):
        text = """
        Deep learning methods have significantly advanced state-of-the-art natural language processing.
        We propose a novel attention mechanism that enhances contextual representations across long sequences.
        """
        vec_a = self.fingerprinter.extract_vector(text)
        vec_b = self.fingerprinter.extract_vector(text)
        res = self.fingerprinter.compare_vectors(vec_a, vec_b)
        
        assert res.distance_score < 0.05
        assert res.burrows_delta == 0.0
        assert res.is_consistent is True

    def test_compare_dissimilar_texts(self):
        academic_text = """
        The empirical investigation demonstrates a statistically significant correlation between parameter scaling
        and convergence rate in dense neural topologies. Furthermore, regularization mitigates catastrophic forgetting.
        """ * 3

        informal_text = """
        Hey guys! So basically I tried this cool thing today and it was super fun and easy.
        I really loved how it worked out and you should definitely try it too!
        """ * 3

        vec_a = self.fingerprinter.extract_vector(academic_text)
        vec_b = self.fingerprinter.extract_vector(informal_text)
        res = self.fingerprinter.compare_vectors(vec_a, vec_b)
        
        assert res.distance_score > 0.15
        assert len(res.divergence_areas) > 0

    def test_analyze_segments(self):
        text_a = "The methodology incorporates convolutional filter banks with residual connections. " * 30
        text_b = "I think that this was totally awesome and really super interesting to do. " * 30
        full_text = text_a + "\n\n" + text_b
        
        results = self.fingerprinter.analyze_segments(full_text, segment_size_words=100)
        assert len(results) >= 1
        assert isinstance(results[0], ComparisonResult)


class TestSentenceAIScoring:

    def test_sentence_scores_and_heatmap_generated(self):
        detector = AIContentDetector()
        # Mocking perplexity to test sentence aggregation logic
        text = "This is a clean academic sentence. It demonstrates clear empirical methodology."
        result = detector.detect(text)
        
        assert hasattr(result, "sentence_scores")
        assert hasattr(result, "sentence_heatmap")
        assert len(result.sentence_scores) >= 1
        
        s = result.sentence_scores[0]
        assert isinstance(s, SentenceAIScore)
        assert s.start_offset >= 0
        assert s.end_offset > s.start_offset
        assert 0.0 <= s.ai_probability <= 1.0

        # Verify heatmap payload format
        assert len(result.sentence_heatmap) == len(result.sentence_scores)
        hm = result.sentence_heatmap[0]
        assert "start" in hm
        assert "end" in hm
        assert "ai_probability" in hm
        assert "verdict" in hm
        assert "color" in hm
