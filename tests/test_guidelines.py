"""Tests for per-venue guideline compliance checking."""

from aegis.detectors.math_formula import MathFormulaChecker
from aegis.detectors.grammar import GrammarLanguageChecker
from aegis.guidelines.checker import GuidelineComplianceChecker
from aegis.guidelines.profiles import (
    DEFAULT_GUIDELINE_VENUES, GUIDELINE_PROFILES, resolve_guideline_profiles,
)

_GRAMMAR = GrammarLanguageChecker(use_spacy=False)
_MATH = MathFormulaChecker()


def _run_check(text: str, venues=None):
    grammar_result = _GRAMMAR.analyze(text)
    math_result = _MATH.analyze("paper.pdf", "pdf", text)
    checker = GuidelineComplianceChecker(
        math_result=math_result, grammar_result=grammar_result,
        full_text=text, word_count=len(text.split()),
    )
    return checker.check_all(venues)


class TestProfileRegistry:

    def test_all_five_venues_present(self):
        assert set(GUIDELINE_PROFILES.keys()) == {"IEEE", "ACM", "BCS", "IET", "ISACA"}

    def test_default_venues_matches_registry(self):
        assert set(DEFAULT_GUIDELINE_VENUES) == set(GUIDELINE_PROFILES.keys())

    def test_resolve_defaults_to_all_five(self):
        profiles = resolve_guideline_profiles(None)
        assert {p.key for p in profiles} == set(DEFAULT_GUIDELINE_VENUES)

    def test_resolve_subset_case_insensitive(self):
        profiles = resolve_guideline_profiles(["ieee", "Isaca"])
        assert {p.key for p in profiles} == {"IEEE", "ISACA"}


class TestScansRunSeparately:

    def test_each_requested_venue_gets_its_own_result(self):
        results = _run_check("A plain sentence with no notable issues here.",
                              ["IEEE", "ACM", "BCS", "IET", "ISACA"])
        assert set(results.keys()) == {"IEEE", "ACM", "BCS", "IET", "ISACA"}
        for venue, res in results.items():
            assert res.venue == venue
            assert res.source_name  # every venue must cite its own source
            assert res.source_url.startswith("http")

    def test_venue_specific_divergence_on_spelling(self):
        # UK spelling should read differently against IEEE/ACM/ISACA (US)
        # vs BCS/IET (UK) -- this is the core "checked separately" property.
        text = "We analyse the colour of the observed behaviour in detail."
        results = _run_check(text, ["IEEE", "BCS"])
        spelling_ieee = next(c for c in results["IEEE"].checks if "Spelling" in c.rule)
        spelling_bcs = next(c for c in results["BCS"].checks if "Spelling" in c.rule)
        assert spelling_ieee.status == "NEEDS_REVIEW"
        assert spelling_bcs.status == "PASS"


class TestContractionRule:

    def test_ieee_flags_contractions_bcs_does_not(self):
        text = "It's clear that we don't need this additional step at all."
        results = _run_check(text, ["IEEE", "BCS"])
        contraction_ieee = next(c for c in results["IEEE"].checks if c.rule == "Contractions")
        contraction_bcs = next(c for c in results["BCS"].checks if c.rule == "Contractions")
        assert contraction_ieee.status == "NEEDS_REVIEW"
        assert contraction_bcs.status == "PASS"


class TestIsacaSpecificRules:

    def test_first_person_flagged_only_for_isaca(self):
        text = "I believe this approach works well for most practitioners."
        results = _run_check(text, ["IEEE", "ISACA"])
        person_ieee = next(c for c in results["IEEE"].checks if "person" in c.rule.lower())
        person_isaca = next(c for c in results["ISACA"].checks if "person" in c.rule.lower())
        assert person_ieee.status == "NOT_ENOUGH_DATA"
        assert person_isaca.status == "NEEDS_REVIEW"

    def test_word_count_only_checked_for_isaca(self):
        text = "Short draft." * 5
        results = _run_check(text, ["IEEE", "ISACA"])
        wc_ieee = next(c for c in results["IEEE"].checks if "word count" in c.rule.lower())
        wc_isaca = next(c for c in results["ISACA"].checks if "word count" in c.rule.lower())
        assert wc_ieee.status == "NOT_ENOUGH_DATA"
        assert wc_isaca.status == "NEEDS_REVIEW"


class TestOverallStatus:

    def test_overall_needs_review_when_any_check_needs_review(self):
        text = "It's a bad example, don't you think?"
        results = _run_check(text, ["IEEE"])
        assert results["IEEE"].overall_status == "NEEDS_REVIEW"

    def test_never_reports_fail(self):
        # These are advisory style checks, never adjudicated pass/fail --
        # overall_status must only ever be one of these three values.
        text = "It's a bad example, don't you think? I really do."
        results = _run_check(text, list(DEFAULT_GUIDELINE_VENUES))
        for res in results.values():
            assert res.overall_status in ("COMPLIANT", "NEEDS_REVIEW", "NOT_ENOUGH_DATA")
            for check in res.checks:
                assert check.status in ("PASS", "NEEDS_REVIEW", "NOT_ENOUGH_DATA")
