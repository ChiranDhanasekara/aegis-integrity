"""
AEGIS Document Editing & Export Module -- v4.0.

Provides:
  - Round-trip DOCX editing (preserving runs, styles, tables, OMML)
  - Native Word Track Changes (w:ins / w:del revision XML)
  - Standalone PDF report generation
"""

from aegis.export.docx_editor import DocxEditor
from aegis.export.docx_tracked_changes import DocxTrackedChangesExporter
from aegis.export.pdf_exporter import PDFReportExporter

__all__ = [
    "DocxEditor",
    "DocxTrackedChangesExporter",
    "PDFReportExporter",
]
