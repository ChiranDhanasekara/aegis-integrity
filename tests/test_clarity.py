"""
Tests for aegis.writing.clarity_scorer -- ClarityScorer.
"""

import pytest
from aegis.writing.clarity_scorer import ClarityScorer, ClarityReport


class TestClarityScorer:

    def setup_method(self):
        self.scorer = ClarityScorer(use_spacy=True)

    # ------------------------------------------------------------------
    # Basic functionality
    # ------------------------------------------------------------------

    def test_analyze_returns_report(self):
        text = ("Machine learning models have transformed data analysis. "
                "These models can process large datasets efficiently. "
                "However, they require significant computational resources.")
        report = self.scorer.analyze(text)
        assert isinstance(report, ClarityReport)
        assert report.total_sentences >= 2
        assert report.overall_clarity_score > 0

    def test_empty_text(self):
        report = self.scorer.analyze("")
        assert report.total_sentences == 0
        # No sentences → no meaningful score; 0.0 is correct default
        assert report.overall_clarity_score == 0.0

    def test_single_sentence(self):
        text = "Deep learning achieves state-of-the-art performance."
        report = self.scorer.analyze(text)
        assert report.total_sentences >= 1
        assert report.sentences[0].word_count > 0
        assert report.sentences[0].fk_grade >= 0

    # ------------------------------------------------------------------
    # Per-sentence metrics
    # ------------------------------------------------------------------

    def test_sentence_fk_grade_computed(self):
        text = "The cat sat on the mat."
        report = self.scorer.analyze(text)
        # FK Grade can be negative for very simple sentences (formula design);
        # the important thing is that it's computed, not that it's >= 0.
        assert report.sentences[0].fk_grade is not None
        assert isinstance(report.sentences[0].fk_grade, float)

    def test_complex_sentence_scored_higher(self):
        simple = "The model works well."
        complex_text = ("The implementation of the convolutional neural network "
                        "architecture demonstrates the utilization of sophisticated "
                        "mathematical optimization techniques for the determination "
                        "of appropriate hyperparameters.")
        report_simple = self.scorer.analyze(simple)
        report_complex = self.scorer.analyze(complex_text)

        if report_simple.sentences and report_complex.sentences:
            assert (report_complex.sentences[0].complexity_score >=
                    report_simple.sentences[0].complexity_score)

    def test_passive_voice_detected_in_sentence(self):
        text = "The experiment was conducted under controlled conditions."
        report = self.scorer.analyze(text)
        assert any(s.has_passive_voice for s in report.sentences)

    def test_nominalization_counted(self):
        text = "The investigation and determination of the optimization results."
        report = self.scorer.analyze(text)
        assert any(s.nominalization_count > 0 for s in report.sentences)

    # ------------------------------------------------------------------
    # Complexity distribution
    # ------------------------------------------------------------------

    def test_complexity_distribution(self):
        text = ("Simple text here. "
                "The implementation of the convolutional neural network "
                "architecture with sophisticated regularization techniques "
                "and hyperparameter optimization demonstrates the significant "
                "advancement in the utilization of deep learning methodologies "
                "for complex pattern recognition tasks.")
        report = self.scorer.analyze(text)
        dist = report.complexity_distribution
        assert "simple" in dist
        assert "moderate" in dist
        assert "complex" in dist
        assert "very_complex" in dist
        total = sum(dist.values())
        assert total == report.total_sentences

    # ------------------------------------------------------------------
    # Repetition detection
    # ------------------------------------------------------------------

    def test_opening_repetition_detected(self):
        text = ("The model achieves high accuracy on benchmark datasets. "
                "The model achieves competitive performance on real-world data. "
                "The model achieves robust results across different domains.")
        report = self.scorer.analyze(text)
        opening_reps = [r for r in report.repetitions
                        if r.repetition_type == "opening"]
        assert len(opening_reps) >= 1, \
            "Should detect repeated sentence openings"

    def test_vocabulary_repetition_detected(self):
        # Two sentences with very high word overlap
        text = ("The neural network model processes images using convolution. "
                "The neural network model processes videos using convolution.")
        report = self.scorer.analyze(text)
        vocab_reps = [r for r in report.repetitions
                      if r.repetition_type == "vocabulary"]
        assert len(vocab_reps) >= 1, \
            "Should detect high vocabulary overlap between sentences"

    def test_no_false_positive_repetition(self):
        text = ("Machine learning has transformed healthcare. "
                "Quantum computing presents new opportunities for cryptography.")
        report = self.scorer.analyze(text)
        assert len(report.repetitions) == 0, \
            "Unrelated sentences should not trigger repetition"

    # ------------------------------------------------------------------
    # Paragraph coherence
    # ------------------------------------------------------------------

    def test_paragraph_coherence_scored(self):
        text = ("Machine learning models have transformed data analysis. "
                "These models can process large datasets efficiently. "
                "However, they require significant computational resources. "
                "Therefore, GPU acceleration is commonly employed.")
        report = self.scorer.analyze(text)
        assert len(report.paragraph_coherence) >= 1
        coh = report.paragraph_coherence[0]
        assert 0 <= coh.overall_coherence <= 1.0
        assert 0 <= coh.topic_consistency <= 1.0
        assert 0 <= coh.transition_score <= 1.0

    def test_coherent_paragraph_scores_high(self):
        text = ("Deep learning has revolutionized image classification. "
                "Furthermore, these methods have been extended to object detection. "
                "Additionally, semantic segmentation benefits from similar architectures. "
                "Consequently, computer vision applications have improved significantly.")
        report = self.scorer.analyze(text)
        if report.paragraph_coherence:
            # Well-connected paragraph should score reasonably
            assert report.paragraph_coherence[0].transition_score > 0.3

    def test_multi_paragraph(self):
        text = ("First paragraph about topic A. This discusses methodology.\n\n"
                "Second paragraph about topic B. This covers the results. "
                "Moreover, the findings are significant.")
        report = self.scorer.analyze(text)
        assert report.total_paragraphs >= 2

    # ------------------------------------------------------------------
    # Overall clarity score
    # ------------------------------------------------------------------

    def test_overall_score_range(self):
        text = ("The model achieves high accuracy. "
                "It processes data efficiently. "
                "Results demonstrate significant improvement.")
        report = self.scorer.analyze(text)
        assert 0 <= report.overall_clarity_score <= 100

    def test_clear_text_scores_higher(self):
        clear = ("The model detects anomalies in real time. "
                 "It uses a simple threshold mechanism. "
                 "Results show high precision and recall.")
        dense = ("The aforementioned implementation of the previously "
                 "discussed convolutional neural network architecture "
                 "for the determination of the optimization of the "
                 "classification of the categorization of the data "
                 "demonstrates the utilization of methodologies. "
                 "The investigation of the examination of the "
                 "implementation demonstrates consideration.")

        report_clear = self.scorer.analyze(clear)
        report_dense = self.scorer.analyze(dense)

        assert report_clear.overall_clarity_score > report_dense.overall_clarity_score

    # ------------------------------------------------------------------
    # Syllable counting
    # ------------------------------------------------------------------

    def test_syllable_count(self):
        assert ClarityScorer._count_syllables("the") == 1
        assert ClarityScorer._count_syllables("model") >= 2
        assert ClarityScorer._count_syllables("implementation") >= 4
        assert ClarityScorer._count_syllables("a") == 1

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_very_short_text(self):
        report = self.scorer.analyze("OK.")
        assert report.total_sentences == 0  # Below 10-char threshold

    def test_no_spacy_fallback(self):
        scorer = ClarityScorer(use_spacy=False)
        text = "The model works well. It achieves high accuracy."
        report = scorer.analyze(text)
        assert report.total_sentences >= 1

    def test_document_level_averages(self):
        text = ("Short sentence. "
                "A somewhat longer sentence with more words in it. "
                "An even longer sentence that contains many more words and "
                "extends further across the line.")
        report = self.scorer.analyze(text)
        assert report.avg_sentence_words > 0
        assert report.avg_fk_grade >= 0
        assert report.avg_fog_index >= 0
