"""
Tests for aegis.writing.rewriter -- AcademicRewriter.
"""

import pytest
from aegis.writing.rewriter import AcademicRewriter, RewriterConfig


class TestAcademicRewriter:

    def setup_method(self):
        self.rewriter = AcademicRewriter(RewriterConfig(
            min_confidence=0.0,  # Accept all for testing
        ))

    # ------------------------------------------------------------------
    # Wordiness
    # ------------------------------------------------------------------

    def test_wordiness_in_order_to(self):
        text = "We used this method in order to improve accuracy."
        ss = self.rewriter.analyze(text)
        wordy = ss.by_category("wordiness")
        assert len(wordy) >= 1
        match = [s for s in wordy if "in order to" in s.original_text.lower()]
        assert match, "Should detect 'in order to'"
        assert match[0].suggested_text.lower() == "to"

    def test_wordiness_due_to_the_fact_that(self):
        text = "The experiment failed due to the fact that the sample was contaminated."
        ss = self.rewriter.analyze(text)
        wordy = ss.by_category("wordiness")
        match = [s for s in wordy if "due to the fact that" in s.original_text.lower()]
        assert match, "Should detect 'due to the fact that'"
        assert match[0].suggested_text.lower() == "because"

    def test_wordiness_is_able_to(self):
        text = "The model is able to detect anomalies in real time."
        ss = self.rewriter.analyze(text)
        wordy = ss.by_category("wordiness")
        match = [s for s in wordy if "is able to" in s.original_text.lower()]
        assert match, "Should detect 'is able to'"
        assert match[0].suggested_text.lower() == "can"

    def test_wordiness_throat_clearing(self):
        text = "It is important to note that the results were significant."
        ss = self.rewriter.analyze(text)
        wordy = ss.by_category("wordiness")
        match = [s for s in wordy
                 if "it is important to note that" in s.original_text.lower()]
        assert match, "Should detect throat-clearing phrase"
        assert match[0].suggested_text == ""  # Suggests removal

    def test_wordiness_a_large_number_of(self):
        text = "A large number of participants were included in the study."
        ss = self.rewriter.analyze(text)
        wordy = ss.by_category("wordiness")
        match = [s for s in wordy if "a large number of" in s.original_text.lower()]
        assert match
        assert match[0].suggested_text.lower() == "many"

    # ------------------------------------------------------------------
    # Nominalizations
    # ------------------------------------------------------------------

    def test_nominalization_utilization(self):
        text = "The utilization of deep learning improved accuracy."
        ss = self.rewriter.analyze(text)
        noms = ss.by_category("nominalization")
        assert len(noms) >= 1
        match = [s for s in noms if "utilization" in s.original_text.lower()]
        assert match, "Should detect 'the utilization of'"
        assert match[0].suggested_text == "use"

    def test_nominalization_investigation(self):
        text = "The investigation of this phenomenon required careful analysis."
        ss = self.rewriter.analyze(text)
        noms = ss.by_category("nominalization")
        match = [s for s in noms if "investigation" in s.original_text.lower()]
        assert match
        assert match[0].suggested_text == "investigate"

    # ------------------------------------------------------------------
    # Contractions
    # ------------------------------------------------------------------

    def test_contraction_detected(self):
        text = "We can't use this approach because it doesn't converge."
        ss = self.rewriter.analyze(text)
        style = ss.by_category("style")
        assert len(style) >= 2
        contractions = [s for s in style if "'" in s.original_text]
        assert contractions, "Should detect contractions"

    def test_contraction_expansion(self):
        text = "The model doesn't converge."
        ss = self.rewriter.analyze(text)
        style = ss.by_category("style")
        match = [s for s in style if s.original_text.lower() == "doesn't"]
        assert match
        assert match[0].suggested_text == "does not"

    # ------------------------------------------------------------------
    # Hedging
    # ------------------------------------------------------------------

    def test_hedging_basically(self):
        text = "The algorithm basically processes each node sequentially."
        ss = self.rewriter.analyze(text)
        hedges = ss.by_category("hedge")
        match = [s for s in hedges if "basically" in s.original_text.lower()]
        assert match, "Should detect 'basically'"

    def test_hedging_sort_of(self):
        text = "The results are sort of consistent with prior work."
        ss = self.rewriter.analyze(text)
        hedges = ss.by_category("hedge")
        match = [s for s in hedges if "sort of" in s.original_text.lower()]
        assert match, "Should detect 'sort of'"

    # ------------------------------------------------------------------
    # Usage errors
    # ------------------------------------------------------------------

    def test_usage_comprised_of(self):
        text = "The dataset is comprised of 1000 images."
        ss = self.rewriter.analyze(text)
        grammar = ss.by_category("grammar")
        match = [s for s in grammar if "comprised of" in s.original_text.lower()]
        assert match
        assert match[0].suggested_text.lower() == "composed of"

    def test_usage_irregardless(self):
        text = "Irregardless of the input, the system adapts."
        ss = self.rewriter.analyze(text)
        grammar = ss.by_category("grammar")
        match = [s for s in grammar if "irregardless" in s.original_text.lower()]
        assert match
        assert match[0].suggested_text.lower() == "regardless"

    def test_usage_could_of(self):
        text = "The authors could of used a larger dataset."
        ss = self.rewriter.analyze(text)
        grammar = ss.by_category("grammar")
        match = [s for s in grammar if "could of" in s.original_text.lower()]
        assert match
        assert match[0].suggested_text.lower() == "could have"

    # ------------------------------------------------------------------
    # Repeated words
    # ------------------------------------------------------------------

    def test_repeated_word(self):
        text = "The the model produces accurate results."
        ss = self.rewriter.analyze(text)
        reps = ss.by_category("repetition")
        assert len(reps) >= 1
        assert reps[0].suggested_text.lower() == "the"

    # ------------------------------------------------------------------
    # Passive voice
    # ------------------------------------------------------------------

    def test_passive_voice_detected(self):
        text = "The experiment was conducted under controlled conditions."
        ss = self.rewriter.analyze(text)
        passive = ss.by_category("passive_voice")
        assert len(passive) >= 1

    # ------------------------------------------------------------------
    # Long sentences
    # ------------------------------------------------------------------

    def test_long_sentence_flagged(self):
        # Build a sentence with > 45 words
        words = ["word"] * 50
        text = "This is a very long sentence that " + " ".join(words) + " at the end."
        rewriter = AcademicRewriter(RewriterConfig(
            max_sentence_words=45,
            min_confidence=0.0,
            fix_wordiness=False,  # Avoid other categories
            fix_contractions=False,
            fix_hedging=False,
            fix_nominalizations=False,
            fix_passive_voice=False,
            fix_usage_errors=False,
            fix_repeated_words=False,
            fix_spelling_consistency=False,
        ))
        ss = rewriter.analyze(text)
        long = ss.by_category("sentence_length")
        assert len(long) >= 1

    # ------------------------------------------------------------------
    # Spelling consistency
    # ------------------------------------------------------------------

    def test_spelling_mix_detected(self):
        text = ("The behaviour of the color model was analysed. "
                "The optimization was favorable to the organised approach.")
        ss = self.rewriter.analyze(text)
        spelling = ss.by_category("spelling")
        assert len(spelling) >= 1, "Should detect mixed UK/US spelling"

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def test_disable_categories(self):
        text = "We can't do this in order to improve the utilization of resources."
        config = RewriterConfig(
            fix_wordiness=False,
            fix_nominalizations=False,
            fix_contractions=False,
            fix_hedging=False,
            fix_passive_voice=False,
            fix_spelling_consistency=False,
            fix_usage_errors=False,
            fix_repeated_words=False,
            suggest_sentence_splits=False,
        )
        rewriter = AcademicRewriter(config)
        ss = rewriter.analyze(text)
        assert len(ss) == 0

    def test_min_confidence_filtering(self):
        text = "The utilization of deep learning improved accuracy."
        # High min_confidence should filter out lower-confidence suggestions
        config = RewriterConfig(min_confidence=0.99)
        rewriter = AcademicRewriter(config)
        ss = rewriter.analyze(text)
        # Nominalizations have confidence=0.65, should be filtered
        noms = ss.by_category("nominalization")
        assert len(noms) == 0

    # ------------------------------------------------------------------
    # Character offsets
    # ------------------------------------------------------------------

    def test_offsets_are_correct(self):
        text = "The model is able to detect anomalies."
        ss = self.rewriter.analyze(text)
        for s in ss:
            # Verify the original_text matches what's at the offset
            actual = text[s.start_offset:s.end_offset]
            assert actual.lower() == s.original_text.lower(), \
                f"Offset mismatch: expected '{s.original_text}' at " \
                f"[{s.start_offset}:{s.end_offset}], got '{actual}'"

    # ------------------------------------------------------------------
    # Empty / edge cases
    # ------------------------------------------------------------------

    def test_empty_text(self):
        ss = self.rewriter.analyze("")
        assert len(ss) == 0

    def test_none_text(self):
        ss = self.rewriter.analyze(None)
        assert len(ss) == 0

    def test_clean_text_no_suggestions(self):
        text = "The model achieves high accuracy on the benchmark dataset."
        ss = self.rewriter.analyze(text)
        # This clean sentence should produce few or no actionable suggestions
        # (maybe passive voice, but that's low confidence)
        high_conf = [s for s in ss if s.confidence > 0.8]
        assert len(high_conf) == 0, \
            f"Clean text should not have high-confidence issues, got: " \
            f"{[s.category for s in high_conf]}"
