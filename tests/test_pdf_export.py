"""
Unit tests for aegis.export.pdf_exporter.
"""

import tempfile
from pathlib import Path
import pytest

from aegis.export.pdf_exporter import PDFReportExporter
from aegis.detectors.similarity_report import SimilarityReport, MatchSpan
from aegis.writing.suggestion import WritingSuggestion, SuggestionSet


class TestPDFReportExporter:

    def test_generate_pdf_summary_report(self):
        exporter = PDFReportExporter()
        
        sim_report = SimilarityReport(
            submission_path="test_manuscript.docx",
            body_text_length=1500,
            matched_char_count=180,
            similarity_percentage=12.0,
            spans=[
                MatchSpan(
                    start_offset=10,
                    end_offset=70,
                    matched_text="Deep convolutional neural network architectures have shown remarkable performance",
                    source_label="IEEE Trans. Medical Imaging (2023)",
                    source_excerpt="Deep convolutional neural network architectures have shown remarkable performance in image segmentation",
                    similarity_score=0.85,
                    match_type="verbatim",
                    detector="ngram",
                )
            ]
        )

        sug_set = SuggestionSet()
        sug_set.add(WritingSuggestion(
            category="wordiness",
            severity="info",
            original_text="in order to",
            suggested_text="to",
            explanation="Shorten for concise academic expression.",
            start_offset=100,
            end_offset=111,
            confidence=0.85,
        ))

        clarity_data = {
            "overall_score": 88,
            "fk_grade": 13.5,
            "fog_index": 14.2,
        }

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_pdf = tmp.name

        try:
            out_path = exporter.generate_report(
                output_path=tmp_pdf,
                doc_title="test_manuscript.docx",
                similarity_report=sim_report,
                suggestions=sug_set,
                clarity_report=clarity_data,
                overall_risk="LOW",
            )
            assert out_path.exists()
            assert out_path.stat().st_size > 1000  # PDF generated with content
            
            # Check with PyMuPDF that PDF is readable and has pages
            import fitz
            doc = fitz.open(str(out_path))
            assert len(doc) >= 1
            page_text = doc[0].get_text()
            assert "AEGIS" in page_text
            assert "test_manuscript.docx" in page_text
            doc.close()
        finally:
            if Path(tmp_pdf).exists():
                Path(tmp_pdf).unlink()
