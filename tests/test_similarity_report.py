"""
Unit tests for aegis.detectors.similarity_report and aegis.corpus.open_access.
"""

import pytest
from unittest.mock import MagicMock, patch

from aegis.detectors.ngram import NGramMatch
from aegis.detectors.semantic import SemanticMatch
from aegis.detectors.similarity_report import (
    MatchSpan,
    SimilarityReport,
    SimilarityReportGenerator,
)
from aegis.corpus.open_access import (
    SourceResult,
    OpenAccessSearchResult,
    OpenAccessCorpusBuilder,
)


class TestSimilarityReportGenerator:

    def setup_method(self):
        self.generator = SimilarityReportGenerator(
            verbatim_threshold=0.50,
            paraphrase_threshold=0.30,
        )

    def test_empty_matches_returns_zero_similarity(self):
        body = "This is a clean academic document with no matching sources."
        report = self.generator.generate(
            body_text=body,
            submission_path="test_doc.docx",
            ngram_matches=[],
            semantic_matches=[],
        )
        assert isinstance(report, SimilarityReport)
        assert report.similarity_percentage == 0.0
        assert report.matched_char_count == 0
        assert len(report.spans) == 0
        assert report.has_matches is False

    def test_ngram_verbatim_match(self):
        body = "Introduction: Deep learning architectures have revolutionized image classification and natural language processing."
        matched_str = "Deep learning architectures have revolutionized image classification"
        match = NGramMatch(
            query_segment=matched_str,
            source_label="paper_2020.pdf",
            source_segment="Deep learning architectures have revolutionized image classification in recent years.",
            jaccard_estimate=0.85,
            match_type="word_ngram",
        )

        report = self.generator.generate(
            body_text=body,
            submission_path="test_doc.docx",
            ngram_matches=[match],
        )

        assert len(report.spans) == 1
        span = report.spans[0]
        assert span.match_type == "verbatim"
        assert span.source_label == "paper_2020.pdf"
        assert span.similarity_score == 0.85
        assert span.start_offset >= 0
        assert span.end_offset > span.start_offset
        assert report.matched_char_count > 0
        assert report.similarity_percentage > 0.0
        assert report.verbatim_match_count == 1

    def test_ngram_paraphrase_match(self):
        body = "We evaluated the convolutional neural network with data augmentation techniques."
        match = NGramMatch(
            query_segment="We evaluated the convolutional neural network",
            source_label="prior_study.pdf",
            source_segment="The authors evaluated the convolutional neural network",
            jaccard_estimate=0.40,  # Below 0.50, above 0.30
            match_type="word_ngram",
        )

        report = self.generator.generate(
            body_text=body,
            submission_path="test_doc.docx",
            ngram_matches=[match],
        )

        assert len(report.spans) == 1
        span = report.spans[0]
        assert span.match_type == "paraphrase"
        assert report.paraphrase_match_count == 1

    def test_semantic_match_conversion(self):
        body = "Neural networks discover latent representations in unstructured datasets."
        match = SemanticMatch(
            query_sentence="Neural networks discover latent representations in unstructured datasets.",
            source_label="deep_repr.pdf",
            source_sentence="Deep neural models automatically extract hidden features from raw data.",
            cosine_score=0.88,
            rerank_score=1.2,
            is_paraphrase=True,
        )

        report = self.generator.generate(
            body_text=body,
            submission_path="test_doc.docx",
            semantic_matches=[match],
        )

        assert len(report.spans) == 1
        span = report.spans[0]
        assert span.match_type == "semantic"
        assert span.detector == "semantic"
        assert span.source_label == "deep_repr.pdf"
        assert report.semantic_match_count == 1

    def test_span_merging_overlapping_takes_higher_score(self):
        body = "This is a sample text segment that has multiple overlapping detections."
        # Span 1: lower score
        s1 = MatchSpan(
            start_offset=0,
            end_offset=30,
            matched_text=body[:30],
            source_label="source_A",
            source_excerpt="excerpt A",
            similarity_score=0.40,
            match_type="paraphrase",
            detector="ngram",
        )
        # Span 2: higher score, overlapping
        s2 = MatchSpan(
            start_offset=10,
            end_offset=35,
            matched_text=body[10:35],
            source_label="source_B",
            source_excerpt="excerpt B",
            similarity_score=0.90,
            match_type="verbatim",
            detector="ngram",
        )

        merged = self.generator._merge_spans([s1, s2])
        assert len(merged) == 1
        assert merged[0].source_label == "source_B"
        assert merged[0].similarity_score == 0.90

    def test_coverage_calculation_multiple_spans(self):
        # Disjoint spans
        spans = [
            MatchSpan(
                start_offset=0, end_offset=10, matched_text="a"*10,
                source_label="s1", source_excerpt="e1", similarity_score=0.8,
                match_type="verbatim", detector="ngram",
            ),
            MatchSpan(
                start_offset=20, end_offset=35, matched_text="b"*15,
                source_label="s2", source_excerpt="e2", similarity_score=0.8,
                match_type="verbatim", detector="ngram",
            ),
        ]
        coverage = self.generator._calculate_coverage(spans)
        assert coverage == 10 + 15

    def test_source_breakdown(self):
        spans = [
            MatchSpan(
                start_offset=0, end_offset=20, matched_text="text1",
                source_label="paper_A.pdf", source_excerpt="e1",
                similarity_score=0.75, match_type="verbatim", detector="ngram",
            ),
            MatchSpan(
                start_offset=30, end_offset=50, matched_text="text2",
                source_label="paper_A.pdf", source_excerpt="e2",
                similarity_score=0.85, match_type="verbatim", detector="ngram",
            ),
            MatchSpan(
                start_offset=60, end_offset=75, matched_text="text3",
                source_label="paper_B.pdf", source_excerpt="e3",
                similarity_score=0.45, match_type="paraphrase", detector="ngram",
            ),
        ]
        breakdown = self.generator._source_breakdown(spans)
        assert "paper_A.pdf" in breakdown
        assert breakdown["paper_A.pdf"]["match_count"] == 2
        assert breakdown["paper_A.pdf"]["max_score"] == 0.85
        assert breakdown["paper_A.pdf"]["total_chars"] == 40
        assert "paper_B.pdf" in breakdown

    def test_to_dict_serialization(self):
        body = "Short sample text for serialization validation."
        match = NGramMatch(
            query_segment="Short sample text",
            source_label="src.pdf",
            source_segment="Short sample text",
            jaccard_estimate=0.9,
            match_type="word_ngram",
        )
        report = self.generator.generate(
            body_text=body,
            submission_path="doc.docx",
            ngram_matches=[match],
        )
        d = report.to_dict()
        assert d["submission_path"] == "doc.docx"
        assert "similarity_percentage" in d
        assert "spans" in d
        assert len(d["spans"]) == 1
        assert d["spans"][0]["source_label"] == "src.pdf"


class TestOpenAccessCorpusBuilder:

    def setup_method(self):
        self.builder = OpenAccessCorpusBuilder(
            use_openalex=False,
            use_semantic_scholar=False,
            use_arxiv=False,
            use_core=False,
        )

    def test_extract_key_phrases(self):
        text = """
        Deep convolutional neural network architectures have shown remarkable performance
        in medical image segmentation and lesion classification. Optimization of convolutional neural
        layers requires extensive dataset curation and regularization techniques.
        """
        phrases = self.builder._extract_key_phrases(text, max_phrases=4)
        assert len(phrases) > 0
        assert any("convolutional" in p or "neural" in p for p in phrases)

    def test_deduplicate(self):
        r1 = SourceResult(
            source_api="arxiv",
            title="Attention Is All You Need",
            authors=["Vaswani et al."],
            year=2017,
            abstract="Transformer model...",
            doi=None,
            url=None,
            relevance_score=0.5,
        )
        r2 = SourceResult(
            source_api="semantic_scholar",
            title="attention is all you need",  # Same title, different casing
            authors=["Vaswani"],
            year=2017,
            abstract="Another abstract...",
            doi=None,
            url=None,
            relevance_score=0.5,
        )
        r3 = SourceResult(
            source_api="openalex",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            authors=["Devlin et al."],
            year=2018,
            abstract="BERT model...",
            doi=None,
            url=None,
            relevance_score=0.5,
        )

        deduped = self.builder._deduplicate([r1, r2, r3])
        assert len(deduped) == 2
        titles = [r.title.lower() for r in deduped]
        assert "attention is all you need" in titles

    def test_parse_openalex_inverted_index(self):
        work = {
            "title": "Sample Work",
            "publication_year": 2021,
            "doi": "https://doi.org/10.1234/567",
            "authorships": [
                {"author": {"display_name": "Alice Smith"}},
                {"author": {"display_name": "Bob Jones"}},
            ],
            "abstract_inverted_index": {
                "This": [0],
                "is": [1],
                "the": [2],
                "abstract": [3],
                "text.": [4],
            },
        }
        res = self.builder._parse_openalex_work(work)
        assert res.title == "Sample Work"
        assert res.abstract == "This is the abstract text."
        assert res.authors == ["Alice Smith", "Bob Jones"]
        assert res.year == 2021

    def test_parse_s2_paper(self):
        paper = {
            "title": "Neural Representation",
            "authors": [{"name": "Carol Danvers"}],
            "year": 2022,
            "abstract": "We present a novel neural architecture.",
            "externalIds": {"DOI": "10.1000/182"},
            "url": "https://www.semanticscholar.org/paper/123",
        }
        res = self.builder._parse_s2_paper(paper)
        assert res.title == "Neural Representation"
        assert res.doi == "10.1000/182"
        assert res.authors == ["Carol Danvers"]

    def test_cache_key_generation(self):
        key1 = self.builder._cache_key("openalex", "quantum computing")
        key2 = self.builder._cache_key("openalex", "quantum computing")
        key3 = self.builder._cache_key("openalex", "machine learning")
        assert key1 == key2
        assert key1 != key3
        assert key1.startswith("openalex_")
