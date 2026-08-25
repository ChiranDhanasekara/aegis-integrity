"""Tests for the grammar & language convention checker."""

from aegis.detectors.grammar import GrammarLanguageChecker

# Disable spaCy in tests so results are deterministic across environments
# that may or may not have en_core_web_sm installed.
_CHECKER = GrammarLanguageChecker(use_spacy=False)


class TestSpellingVariant:

    def test_detects_consistent_us_spelling(self):
        result = _CHECKER.analyze("We analyze the color of the behavior observed.")
        assert result.spelling_variant_detected == "US"

    def test_detects_consistent_uk_spelling(self):
        result = _CHECKER.analyze("We analyse the colour of the behaviour observed.")
        assert result.spelling_variant_detected == "UK"

    def test_detects_mixed_spelling(self):
        result = _CHECKER.analyze("We analyse the color of the behaviour observed.")
        assert result.spelling_variant_detected == "MIXED"
        mixed_issues = [i for i in result.issues if i.category == "spelling_mix"]
        assert len(mixed_issues) == 1
        assert mixed_issues[0].severity == "MEDIUM"

    def test_unknown_when_no_marker_words_present(self):
        result = _CHECKER.analyze("This sentence contains no marker spellings at all.")
        assert result.spelling_variant_detected == "UNKNOWN"


class TestContractions:

    def test_contractions_counted(self):
        result = _CHECKER.analyze("It's clear that we don't need this and can't skip it.")
        assert result.contraction_count == 3
        assert any(i.category == "contraction" for i in result.issues)

    def test_idiomatic_dont_care_excluded(self):
        result = _CHECKER.analyze("This is a classic don't-care condition in the truth table.")
        assert result.contraction_count == 0

    def test_no_contractions_in_formal_text(self):
        result = _CHECKER.analyze("It is clear that we do not need this and cannot skip it.")
        assert result.contraction_count == 0


class TestAgreementHeuristics:

    def test_data_is_flagged(self):
        result = _CHECKER.analyze("The data is collected from ten sensors over one year.")
        assert any(i.category == "agreement" for i in result.issues)

    def test_data_are_not_flagged(self):
        result = _CHECKER.analyze("The data are collected from ten sensors over one year.")
        agreement_issues = [i for i in result.issues if i.category == "agreement"]
        assert agreement_issues == []

    def test_a_series_of_singular_ok(self):
        result = _CHECKER.analyze("A series of tests was run to validate the hypothesis.")
        agreement_issues = [i for i in result.issues if i.category == "agreement"]
        assert agreement_issues == []

    def test_a_series_of_plural_flagged(self):
        result = _CHECKER.analyze("A series of tests were run to validate the hypothesis.")
        assert any(i.category == "agreement" for i in result.issues)


class TestUsageChecks:

    def test_comprised_of_flagged(self):
        result = _CHECKER.analyze("The system is comprised of three modules.")
        assert any("comprised of" in i.message for i in result.issues)

    def test_modal_of_flagged(self):
        result = _CHECKER.analyze("We could of tested this earlier in the process.")
        assert any("modal" in i.message for i in result.issues)

    def test_less_plus_countable_noun_flagged(self):
        result = _CHECKER.analyze("We observed less errors than in the baseline run.")
        assert any("fewer" in i.message for i in result.issues)

    def test_decade_apostrophe_flagged(self):
        result = _CHECKER.analyze("Sensors from the 1990's era were used in this study.")
        assert any("decade" in i.message for i in result.issues)

    def test_acronym_plural_apostrophe_flagged(self):
        result = _CHECKER.analyze("The FET's are placed across the substrate.")
        assert any("acronym plural" in i.message for i in result.issues)

    def test_acronym_possessive_not_flagged(self):
        # "IEEE's policy" is a correct possessive, not a plural -- must not
        # be flagged by the acronym-plural-apostrophe check.
        result = _CHECKER.analyze("IEEE's policy requires double-blind review.")
        assert not any("acronym plural" in i.message for i in result.issues)

    def test_repeated_word_flagged(self):
        result = _CHECKER.analyze("This is is a simple duplication error.")
        assert any(i.category == "usage" and "repeated" in i.message for i in result.issues)


class TestQualityScore:

    def test_clean_text_scores_near_one(self):
        result = _CHECKER.analyze(
            "This paper presents a clear and well-structured evaluation. "
            "The methodology is described in detail, and the results are "
            "consistent with prior work in the field."
        )
        assert result.quality_score >= 0.9

    def test_empty_text_scores_one(self):
        result = _CHECKER.analyze("")
        assert result.quality_score == 1.0
        assert result.word_count == 0
