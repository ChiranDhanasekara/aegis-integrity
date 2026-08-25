"""
GuidelineComplianceChecker -- applies math + grammar findings against one
venue's GuidelineProfile at a time. See aegis/guidelines/__init__.py for
why venues are checked separately rather than merged into one score.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

from aegis.detectors.math_formula import MathAnalysisResult
from aegis.detectors.grammar import GrammarAnalysisResult
from aegis.guidelines.profiles import GuidelineProfile, resolve_guideline_profiles

_NUMERIC_CITATION_RE = re.compile(r"\[\d+(?:\s*[,–-]\s*\d+)*\]")
_AUTHOR_YEAR_CITATION_RE = re.compile(
    r"\([A-Z][A-Za-z\-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z\-]+|\s+et\s+al\.?)?,?\s+(?:19|20)\d{2}\)"
)


@dataclass
class ComplianceCheck:
    rule: str
    status: str        # PASS | NEEDS_REVIEW | NOT_ENOUGH_DATA
    detail: str
    source: str


@dataclass
class GuidelineComplianceResult:
    venue: str
    display_name: str
    source_name: str
    source_url: str
    checks: list[ComplianceCheck] = field(default_factory=list)
    overall_status: str = "NOT_ENOUGH_DATA"   # COMPLIANT | NEEDS_REVIEW | NOT_ENOUGH_DATA

    @property
    def needs_review_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "NEEDS_REVIEW")


class GuidelineComplianceChecker:
    """
    Runs one venue's GuidelineProfile against already-computed math and
    grammar findings plus a few direct text scans (citation-marker style,
    word count, list density). Never touches plagiarism/AI/citation risk.
    """

    def __init__(
        self,
        math_result: Optional[MathAnalysisResult],
        grammar_result: Optional[GrammarAnalysisResult],
        full_text: str,
        word_count: int,
    ):
        self.math = math_result
        self.grammar = grammar_result
        self.full_text = full_text or ""
        self.word_count = word_count

    def check_all(self, venues: Optional[list[str]] = None) -> dict[str, GuidelineComplianceResult]:
        return {p.key: self.check(p) for p in resolve_guideline_profiles(venues)}

    def check(self, profile: GuidelineProfile) -> GuidelineComplianceResult:
        checks: list[ComplianceCheck] = []
        checks.append(self._check_spelling(profile))
        checks.append(self._check_contractions(profile))
        checks.append(self._check_person(profile))
        checks.append(self._check_equation_style(profile))
        checks.append(self._check_notation_conventions(profile))
        checks.append(self._check_citation_marker_style(profile))
        checks.append(self._check_word_count(profile))
        checks.append(self._check_list_density(profile))

        if any(c.status == "NEEDS_REVIEW" for c in checks):
            overall = "NEEDS_REVIEW"
        elif any(c.status == "PASS" for c in checks):
            overall = "COMPLIANT"
        else:
            overall = "NOT_ENOUGH_DATA"

        return GuidelineComplianceResult(
            venue=profile.key, display_name=profile.display_name,
            source_name=profile.source_name, source_url=profile.source_url,
            checks=checks, overall_status=overall,
        )

    # ------------------------------------------------------------------

    def _check_spelling(self, profile: GuidelineProfile) -> ComplianceCheck:
        rule = f"Spelling variant ({profile.spelling_variant} English)"
        source = profile.source_name + (
            "" if profile.spelling_variant_confidence == "stated"
            else " (inferred from publisher, not explicitly stated)"
        )
        if not self.grammar or self.grammar.spelling_variant_detected == "UNKNOWN":
            return ComplianceCheck(rule, "NOT_ENOUGH_DATA",
                                    "No UK/US-marker spellings were found to classify.", source)
        detected = self.grammar.spelling_variant_detected
        if detected == "MIXED":
            return ComplianceCheck(
                rule, "NEEDS_REVIEW",
                f"Document mixes British and American spelling "
                f"({self.grammar.spelling_variant_counts}); {profile.display_name} "
                f"expects consistent {profile.spelling_variant} English.",
                source,
            )
        if detected != profile.spelling_variant:
            confidence_note = (
                " (this expectation is inferred, not explicitly published -- "
                "treat as a suggestion)" if profile.spelling_variant_confidence == "inferred" else ""
            )
            return ComplianceCheck(
                rule, "NEEDS_REVIEW",
                f"Document consistently uses {detected} English spelling; "
                f"{profile.display_name} expects {profile.spelling_variant} "
                f"English{confidence_note}.",
                source,
            )
        return ComplianceCheck(rule, "PASS",
                                f"Consistently uses {profile.spelling_variant} English.", source)

    def _check_contractions(self, profile: GuidelineProfile) -> ComplianceCheck:
        rule = "Contractions"
        source = profile.source_name
        if profile.allow_contractions:
            return ComplianceCheck(rule, "PASS",
                                    f"{profile.display_name} does not restrict contractions.", source)
        if not self.grammar:
            return ComplianceCheck(rule, "NOT_ENOUGH_DATA", "Grammar check did not run.", source)
        if self.grammar.contraction_count > 0:
            return ComplianceCheck(
                rule, "NEEDS_REVIEW",
                f"{self.grammar.contraction_count} contraction(s) found; "
                f"{profile.display_name} avoids contractions in formal text.",
                source,
            )
        return ComplianceCheck(rule, "PASS", "No contractions found.", source)

    def _check_person(self, profile: GuidelineProfile) -> ComplianceCheck:
        rule = "First/second person address"
        source = profile.source_name
        if not profile.disallow_first_person_singular:
            return ComplianceCheck(rule, "NOT_ENOUGH_DATA",
                                    f"{profile.display_name}'s published guidelines do not "
                                    f"mandate a particular grammatical person.", source)
        if not self.grammar:
            return ComplianceCheck(rule, "NOT_ENOUGH_DATA", "Grammar check did not run.", source)
        hits = self.grammar.first_person_singular_count + self.grammar.second_person_count
        if hits:
            return ComplianceCheck(
                rule, "NEEDS_REVIEW",
                f'{self.grammar.first_person_singular_count} use(s) of "I" and '
                f'{self.grammar.second_person_count} use(s) of "you"/"your" found; '
                f'{profile.display_name} requires third person ("avoid the use of '
                f'\'I\' or \'you\'").',
                source,
            )
        return ComplianceCheck(rule, "PASS", 'No first/second-person address found.', source)

    def _check_equation_style(self, profile: GuidelineProfile) -> ComplianceCheck:
        rule = "Equation reference style"
        source = profile.source_name
        if not self.math or self.math.equations_found == 0:
            return ComplianceCheck(rule, "NOT_ENOUGH_DATA",
                                    "No numbered equations were found in this document.", source)
        if profile.equation_ref_style is None:
            return ComplianceCheck(rule, "NOT_ENOUGH_DATA", profile.equation_ref_note, source)
        ref_issues = [i for i in self.math.reference_issues if "phrasing" in i.message.lower()]
        if profile.equation_ref_style == "bare-parens" and ref_issues:
            return ComplianceCheck(
                rule, "NEEDS_REVIEW",
                f"{profile.equation_ref_note} Equation references in this document "
                f"use inconsistent or non-bare phrasing.",
                source,
            )
        numbering_problems = [i for i in self.math.numbering_issues if i.severity == "MEDIUM"]
        if numbering_problems:
            return ComplianceCheck(
                rule, "NEEDS_REVIEW",
                f"{len(numbering_problems)} equation numbering issue(s) found "
                f"(duplicates or out-of-order numbers).",
                source,
            )
        return ComplianceCheck(rule, "PASS", f"{profile.equation_ref_note} No issues found.", source)

    def _check_notation_conventions(self, profile: GuidelineProfile) -> ComplianceCheck:
        rule = "Numeric/notation conventions"
        source = profile.source_name
        if not self.math or not self.math.notation_issues:
            if not self.math or self.math.equations_found == 0:
                return ComplianceCheck(rule, "NOT_ENOUGH_DATA", "No formulas found to check.", source)
            return ComplianceCheck(rule, "PASS", "No notation issues found.", source)

        relevant = self.math.notation_issues
        if profile.key != "IET":
            relevant = [i for i in relevant if "exponential" not in i.message.lower()]
        if profile.key != "IEEE":
            relevant = [i for i in relevant
                        if "percentage" not in i.message.lower()
                        and "leading zero" not in i.message.lower()
                        and "doubly-parenthesised" not in i.message.lower()]
        if not relevant:
            return ComplianceCheck(rule, "NOT_ENOUGH_DATA",
                                    "No venue-specific notation conventions to check for "
                                    "this document.", source)
        return ComplianceCheck(
            rule, "NEEDS_REVIEW",
            "; ".join(i.message for i in relevant),
            source,
        )

    def _check_citation_marker_style(self, profile: GuidelineProfile) -> ComplianceCheck:
        rule = "In-text citation marker style"
        source = profile.source_name
        if profile.citation_style not in ("numeric-bracket",):
            return ComplianceCheck(rule, "NOT_ENOUGH_DATA",
                                    f"{profile.display_name} uses {profile.citation_style} "
                                    f"citations, which this scan does not verify.", source)
        numeric_hits = len(_NUMERIC_CITATION_RE.findall(self.full_text))
        author_year_hits = len(_AUTHOR_YEAR_CITATION_RE.findall(self.full_text))
        if numeric_hits == 0 and author_year_hits == 0:
            return ComplianceCheck(rule, "NOT_ENOUGH_DATA",
                                    "No recognizable in-text citation markers found.", source)
        if author_year_hits > numeric_hits:
            return ComplianceCheck(
                rule, "NEEDS_REVIEW",
                f"Citations appear to predominantly use author-year style "
                f"({author_year_hits} marker(s)) rather than numeric brackets "
                f"({numeric_hits}); {profile.display_name} expects numeric-bracket "
                f'citations, e.g. "[1]".',
                source,
            )
        return ComplianceCheck(rule, "PASS",
                                f"Predominantly numeric-bracket citations "
                                f"({numeric_hits} marker(s)).", source)

    def _check_word_count(self, profile: GuidelineProfile) -> ComplianceCheck:
        rule = "Target word count"
        source = profile.source_name
        if profile.word_count_range is None:
            return ComplianceCheck(rule, "NOT_ENOUGH_DATA",
                                    f"{profile.display_name} publishes no fixed word-count "
                                    f"target in the sourced guidelines.", source)
        lo, hi = profile.word_count_range
        if lo <= self.word_count <= hi:
            return ComplianceCheck(rule, "PASS",
                                    f"{self.word_count} words, within the "
                                    f"{lo}-{hi} target range.", source)
        return ComplianceCheck(
            rule, "NEEDS_REVIEW",
            f"{self.word_count} words, outside the {lo}-{hi} target range "
            f"published for {profile.display_name}.",
            source,
        )

    def _check_list_density(self, profile: GuidelineProfile) -> ComplianceCheck:
        rule = "Bulleted-list usage"
        source = profile.source_name
        if not profile.discourage_bulleted_lists:
            return ComplianceCheck(rule, "NOT_ENOUGH_DATA",
                                    f"{profile.display_name} does not publish a stance on "
                                    f"list usage.", source)
        if not self.grammar or self.grammar.paragraph_count == 0:
            return ComplianceCheck(rule, "NOT_ENOUGH_DATA", "No paragraph data available.", source)
        if self.grammar.list_line_count > self.grammar.paragraph_count:
            return ComplianceCheck(
                rule, "NEEDS_REVIEW",
                f"{self.grammar.list_line_count} bulleted/numbered list line(s) "
                f"vs. {self.grammar.paragraph_count} paragraph(s); "
                f'{profile.display_name} asks that "the majority of the article '
                f'consist of paragraphs."',
                source,
            )
        return ComplianceCheck(rule, "PASS", "Prose paragraphs dominate over list items.", source)
