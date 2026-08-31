"""
Word Revision XML & Tracked Changes Exporter -- AEGIS v4.0.

Exports academic manuscripts with native Microsoft Word tracked changes (<w:ins> and <w:del> tags).
When opened in Microsoft Word or LibreOffice, changes appear as formal revision redlines with
author attribution ("AEGIS Writing Assistant") and timestamps.
"""

from __future__ import annotations
import datetime
import logging
from pathlib import Path
from typing import Optional, Union

import docx
from docx.document import Document
from docx.text.paragraph import Paragraph
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

from aegis.writing.suggestion import SuggestionSet, WritingSuggestion

logger = logging.getLogger(__name__)

# WordprocessingML XML Namespaces
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


class DocxTrackedChangesExporter:
    """
    Export documents with native Word Track Changes / revisions.

    Usage::

        exporter = DocxTrackedChangesExporter("draft.docx", author="AEGIS Writing Assistant")
        exporter.apply_tracked_suggestions(suggestion_set)
        exporter.save("draft_with_tracked_changes.docx")
    """

    def __init__(
        self,
        docx_path_or_doc: Optional[Union[str, Path, Document]] = None,
        author: str = "AEGIS Writing Assistant",
    ):
        self.author = author
        self._revision_id = 1
        
        if docx_path_or_doc is None:
            self._doc = docx.Document()
            self._source_path = None
        elif isinstance(docx_path_or_doc, Document):
            self._doc = docx_path_or_doc
            self._source_path = None
        else:
            path = Path(docx_path_or_doc)
            if not path.exists():
                raise FileNotFoundError(f"DOCX file not found: {path}")
            self._doc = docx.Document(str(path))
            self._source_path = path

    @property
    def document(self) -> Document:
        return self._doc

    def _next_revision_id(self) -> int:
        curr = self._revision_id
        self._revision_id += 1
        return curr

    def _get_timestamp(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def insert_tracked_change(
        self,
        paragraph: Paragraph,
        original_text: str,
        suggested_text: str,
    ) -> bool:
        """
        Insert a tracked deletion (<w:del>) and insertion (<w:ins>) into a paragraph.
        """
        if original_text not in paragraph.text:
            return False

        now_str = self._get_timestamp()
        del_id = self._next_revision_id()
        ins_id = self._next_revision_id()

        # Build XML for deletion
        del_xml = (
            f'<w:del {nsdecls("w")} w:id="{del_id}" w:author="{self.author}" w:date="{now_str}">'
            f'<w:r><w:delText xml:space="preserve">{original_text}</w:delText></w:r>'
            f'</w:del>'
        )

        # Build XML for insertion (if not empty removal)
        ins_xml = ""
        if suggested_text:
            ins_xml = (
                f'<w:ins {nsdecls("w")} w:id="{ins_id}" w:author="{self.author}" w:date="{now_str}">'
                f'<w:r><w:t xml:space="preserve">{suggested_text}</w:t></w:r>'
                f'</w:ins>'
            )

        p_element = paragraph._p

        # Find target in paragraph text
        full_text = paragraph.text
        if original_text in full_text:
            # Reconstruct paragraph XML with revisions
            # Split before and after match
            parts = full_text.split(original_text, 1)
            before_text = parts[0]
            after_text = parts[1]

            # Clear existing children of paragraph except paragraph properties (<w:pPr>)
            p_pr = p_element.pPr
            p_element.clear_content()
            if p_pr is not None:
                p_element.append(p_pr)

            # 1. Before text run
            if before_text:
                r_before = OxmlElement('w:r')
                t_before = OxmlElement('w:t')
                t_before.text = before_text
                t_before.set(qn('xml:space'), 'preserve')
                r_before.append(t_before)
                p_element.append(r_before)

            # 2. Tracked deletion
            del_elem = parse_xml(del_xml)
            p_element.append(del_elem)

            # 3. Tracked insertion
            if ins_xml:
                ins_elem = parse_xml(ins_xml)
                p_element.append(ins_elem)

            # 4. After text run
            if after_text:
                r_after = OxmlElement('w:r')
                t_after = OxmlElement('w:t')
                t_after.text = after_text
                t_after.set(qn('xml:space'), 'preserve')
                r_after.append(t_after)
                p_element.append(r_after)

            return True

        return False

    def apply_tracked_suggestions(
        self, suggestions: Union[SuggestionSet, list[WritingSuggestion]]
    ) -> int:
        """
        Apply accepted writing suggestions as tracked revisions.

        Returns total count of tracked revisions applied.
        """
        if isinstance(suggestions, SuggestionSet):
            to_apply = suggestions.accepted
        else:
            to_apply = [s for s in suggestions if s.status in ("accepted", "modified")]

        applied_count = 0

        for s in to_apply:
            target = s.original_text
            replacement = s.final_text
            
            # Search across paragraphs
            for p in self._doc.paragraphs:
                if target in p.text:
                    if self.insert_tracked_change(p, target, replacement):
                        applied_count += 1
                        break

            # Search in table cells
            for table in self._doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        for p in cell.paragraphs:
                            if target in p.text:
                                if self.insert_tracked_change(p, target, replacement):
                                    applied_count += 1
                                    break

        return applied_count

    def save(self, output_path: Union[str, Path]) -> Path:
        """Save document with tracked changes to disk."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._doc.save(str(out))
        return out
