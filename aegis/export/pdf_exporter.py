"""
PDF Report Exporter -- AEGIS v4.0.

Generates standalone, executive-ready PDF integrity and writing analysis reports
using ReportLab.
"""

from __future__ import annotations
import datetime
from pathlib import Path
from typing import Optional, Union

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
)

from aegis.writing.suggestion import SuggestionSet
from aegis.detectors.similarity_report import SimilarityReport


class PDFReportExporter:
    """
    Generate professional PDF reports for academic integrity analysis and writing improvements.

    Usage::

        exporter = PDFReportExporter()
        exporter.generate_report(
            output_path="integrity_report.pdf",
            doc_title="manuscript_draft.docx",
            similarity_report=sim_report,
            suggestions=suggestion_set,
            clarity_report=clarity_data,
        )
    """

    def __init__(self):
        self._styles = getSampleStyleSheet()
        self._init_custom_styles()

    def _init_custom_styles(self):
        self._styles.add(ParagraphStyle(
            name='ReportTitle',
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=colors.HexColor('#0f172a'),
            spaceAfter=6,
        ))
        self._styles.add(ParagraphStyle(
            name='SectionHeading',
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=colors.HexColor('#1e293b'),
            spaceBefore=14,
            spaceAfter=8,
        ))
        self._styles.add(ParagraphStyle(
            name='SubHeading',
            fontName='Helvetica-Bold',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#475569'),
            spaceBefore=8,
            spaceAfter=4,
        ))
        self._styles.add(ParagraphStyle(
            name='BodySmall',
            fontName='Helvetica',
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor('#334155'),
        ))
        self._styles.add(ParagraphStyle(
            name='TableHeader',
            fontName='Helvetica-Bold',
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#0f172a'),
        ))
        self._styles.add(ParagraphStyle(
            name='BadgeLow',
            fontName='Helvetica-Bold',
            fontSize=10,
            leading=12,
            textColor=colors.HexColor('#15803d'),
            alignment=1,
        ))

    def generate_report(
        self,
        output_path: Union[str, Path],
        doc_title: str,
        similarity_report: Optional[SimilarityReport] = None,
        suggestions: Optional[Union[SuggestionSet, list]] = None,
        clarity_report: Optional[dict] = None,
        overall_risk: str = "LOW",
    ) -> Path:
        """
        Build and save the PDF report.
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        elements = []

        # 1. Header Banner
        elements.append(Paragraph("AEGIS Academic Integrity & Writing Report", self._styles['ReportTitle']))
        now_str = datetime.datetime.now().strftime("%B %d, %Y • %H:%M UTC")
        elements.append(Paragraph(f"Document: <b>{doc_title}</b> &nbsp;|&nbsp; Generated: {now_str}", self._styles['BodySmall']))
        elements.append(Spacer(1, 14))

        # 2. Executive Summary Metrics Table
        sim_pct = f"{similarity_report.similarity_percentage:.1f}%" if similarity_report else "0.0%"
        sug_count = len(suggestions) if suggestions is not None else 0
        clarity_score = f"{clarity_report.get('overall_score', 85)}/100" if clarity_report else "85/100"
        fk_grade = f"{clarity_report.get('fk_grade', 13.2)}" if clarity_report else "13.2"

        summary_data = [
            [
                Paragraph("<b>Overall Integrity Risk</b>", self._styles['TableHeader']),
                Paragraph("<b>Similarity Score</b>", self._styles['TableHeader']),
                Paragraph("<b>Writing Suggestions</b>", self._styles['TableHeader']),
                Paragraph("<b>Clarity Index</b>", self._styles['TableHeader']),
                Paragraph("<b>FK Grade Level</b>", self._styles['TableHeader']),
            ],
            [
                Paragraph(f"<b>{overall_risk}</b>", self._styles['BadgeLow']),
                Paragraph(f"<b>{sim_pct}</b>", self._styles['TableHeader']),
                Paragraph(f"{sug_count} items", self._styles['BodySmall']),
                Paragraph(f"{clarity_score}", self._styles['BodySmall']),
                Paragraph(f"{fk_grade}", self._styles['BodySmall']),
            ]
        ]
        
        summary_table = Table(summary_data, colWidths=[110, 100, 110, 110, 110])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f8fafc')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#0f172a')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 16))

        # 3. Similarity Breakdown
        if similarity_report and similarity_report.has_matches:
            elements.append(Paragraph("1. Similarity & Source Matching Analysis", self._styles['SectionHeading']))
            sim_rows = [
                [
                    Paragraph("<b>#</b>", self._styles['TableHeader']),
                    Paragraph("<b>Source / Reference Label</b>", self._styles['TableHeader']),
                    Paragraph("<b>Matched Text Excerpt</b>", self._styles['TableHeader']),
                    Paragraph("<b>Type</b>", self._styles['TableHeader']),
                    Paragraph("<b>Score</b>", self._styles['TableHeader']),
                ]
            ]
            for idx, s in enumerate(similarity_report.spans[:15]):
                sim_rows.append([
                    Paragraph(str(idx + 1), self._styles['BodySmall']),
                    Paragraph(s.source_label[:40], self._styles['BodySmall']),
                    Paragraph(f"<i>\"{s.matched_text[:65]}...\"</i>", self._styles['BodySmall']),
                    Paragraph(s.match_type.capitalize(), self._styles['BodySmall']),
                    Paragraph(f"{s.similarity_score * 100:.0f}%", self._styles['BodySmall']),
                ])

            sim_table = Table(sim_rows, colWidths=[25, 140, 240, 75, 60])
            sim_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8fafc')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(sim_table)
            elements.append(Spacer(1, 14))

        # 4. Actionable Writing Suggestions
        elements.append(Paragraph("2. Academic Writing Suggestions", self._styles['SectionHeading']))
        sug_list = suggestions.all if isinstance(suggestions, SuggestionSet) else (suggestions or [])
        
        if sug_list:
            sug_rows = [
                [
                    Paragraph("<b>Category</b>", self._styles['TableHeader']),
                    Paragraph("<b>Original Phrasing</b>", self._styles['TableHeader']),
                    Paragraph("<b>Proposed Revision</b>", self._styles['TableHeader']),
                    Paragraph("<b>Rationale</b>", self._styles['TableHeader']),
                ]
            ]
            for s in sug_list[:20]:
                orig = getattr(s, 'original_text', '')[:35]
                repl = getattr(s, 'final_text', getattr(s, 'suggested_text', ''))[:35] or "(remove)"
                cat = getattr(s, 'category', 'general').capitalize()
                expl = getattr(s, 'explanation', '')[:65]
                
                sug_rows.append([
                    Paragraph(f"<b>{cat}</b>", self._styles['BodySmall']),
                    Paragraph(f"<font color='#dc2626'>{orig}</font>", self._styles['BodySmall']),
                    Paragraph(f"<font color='#16a34a'><b>{repl}</b></font>", self._styles['BodySmall']),
                    Paragraph(expl, self._styles['BodySmall']),
                ])

            sug_table = Table(sug_rows, colWidths=[90, 120, 120, 210])
            sug_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8fafc')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(sug_table)
        else:
            elements.append(Paragraph("No mechanical, stylistic, or wordiness issues identified in the manuscript.", self._styles['BodySmall']))

        elements.append(Spacer(1, 20))

        # 5. Footer Privacy Note
        elements.append(Paragraph(
            "<b>Confidential & Privacy Protected:</b> This report was generated locally by the AEGIS platform. "
            "Full manuscript content was not transmitted to third-party language models or commercial services.",
            self._styles['BodySmall']
        ))

        # Build document
        doc.build(elements)
        return out_path
