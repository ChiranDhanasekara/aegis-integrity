"""
Unit tests for aegis.web and web API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from aegis.api.app import app
from aegis.web.serve import create_web_app


class TestWebAPI:

    @classmethod
    def setup_class(cls):
        cls.client = TestClient(app)

    def test_root_serves_html(self):
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "AEGIS" in resp.text

    def test_static_css_served(self):
        resp = self.client.get("/static/css/style.css")
        assert resp.status_code == 200
        assert "--brand-gradient" in resp.text

    def test_static_js_served(self):
        resp = self.client.get("/static/js/app.js")
        assert resp.status_code == 200
        assert "AEGIS Academic Writing" in resp.text

    def test_writing_analyze_endpoint(self):
        text = "In order to improve performance, the model is able to detect errors."
        resp = self.client.post("/api/writing/analyze", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert "suggestions" in data
        assert "clarity" in data
        assert len(data["suggestions"]) >= 2
        # Check 'in order to' and 'is able to' detected
        categories = [s["category"] for s in data["suggestions"]]
        assert "wordiness" in categories
        assert data["clarity"]["overall_score"] > 0

    def test_writing_apply_endpoint(self):
        text = "We used this in order to start."
        payload = {
            "text": text,
            "accepted_ids": ["sug_1"],
            "suggestions": [
                {
                    "id": "sug_1",
                    "category": "wordiness",
                    "severity": "info",
                    "original_text": "in order to",
                    "suggested_text": "to",
                    "explanation": "Shorten.",
                    "start_offset": 13,
                    "end_offset": 24,
                    "confidence": 0.85,
                }
            ],
        }
        resp = self.client.post("/api/writing/apply", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["updated_text"] == "We used this to start."

    def test_document_upload_endpoint(self):
        # Test uploading a text document to /api/document/upload
        files = {"file": ("test_doc.txt", b"In order to test the pipeline, we conduct experiments.")}
        resp = self.client.post("/api/document/upload", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert "suggestions" in data
        assert "clarity" in data
        assert len(data["suggestions"]) >= 1

    def test_export_docx_and_pdf_endpoints(self):
        # Test DOCX export
        resp = self.client.post(
            "/api/export/docx",
            json={"text": "This is a test paragraph.", "tracked_changes": False}
        )
        assert resp.status_code == 200
        assert "openxmlformats" in resp.headers.get("content-type", "")
        assert len(resp.content) > 100

        # Test PDF export
        resp_pdf = self.client.post(
            "/api/export/pdf",
            json={"text": "This is a test paragraph.", "doc_title": "Test Paper"}
        )
        assert resp_pdf.status_code == 200
        assert "application/pdf" in resp_pdf.headers.get("content-type", "")
        assert len(resp_pdf.content) > 500

    def test_standalone_web_app_factory(self):
        web_app = create_web_app()
        web_client = TestClient(web_app)
        resp = web_client.get("/")
        assert resp.status_code == 200
        assert "AEGIS" in resp.text
