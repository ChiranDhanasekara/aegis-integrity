"""
AEGIS Web Server & API Mount.

Serves the interactive web interface and handles frontend writing/clarity API requests.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from aegis.writing.rewriter import AcademicRewriter, RewriterConfig
from aegis.writing.clarity_scorer import ClarityScorer
from aegis.writing.suggestion import SuggestionSet, WritingSuggestion

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"


class TextAnalysisRequest(BaseModel):
    text: str
    spelling_convention: Optional[str] = "auto"
    max_sentence_words: Optional[int] = 45


class ApplySuggestionsRequest(BaseModel):
    text: str
    accepted_ids: list[str]
    suggestions: list[dict]


def create_web_app() -> FastAPI:
    """Create and configure the FastAPI web application."""
    app = FastAPI(
        title="AEGIS Academic Writing & Integrity Platform",
        description="Interactive Web Application & Writing Assistant",
        version="4.0.0",
    )

    # Initialize rule-based engines (offline, lightweight)
    rewriter = AcademicRewriter()
    clarity_scorer = ClarityScorer()

    # Mount static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def serve_index():
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            return HTMLResponse("<h1>AEGIS Web UI (Index file not found)</h1>", status_code=404)
        return FileResponse(index_path)

    @app.post("/api/writing/analyze")
    async def analyze_writing(req: TextAnalysisRequest):
        """Run fast rule-based writing assistance and clarity scoring."""
        if not req.text.strip():
            return {"suggestions": [], "clarity": None}

        # 1. Writing suggestions
        suggestions_set = rewriter.analyze(req.text)
        
        # 2. Clarity metrics
        clarity_report = clarity_scorer.analyze(req.text)

        return {
            "suggestions": [s.to_dict() for s in suggestions_set.all],
            "summary": suggestions_set.summary,
            "clarity": {
                "overall_score": clarity_report.overall_clarity_score,
                "fk_grade": clarity_report.avg_fk_grade,
                "fog_index": clarity_report.avg_fog_index,
                "avg_sentence_words": clarity_report.avg_sentence_words,
                "total_sentences": clarity_report.total_sentences,
                "coherence": round(
                    sum(p.overall_coherence for p in clarity_report.paragraph_coherence) /
                    max(len(clarity_report.paragraph_coherence), 1), 2
                ) if clarity_report.paragraph_coherence else 0.8,
            },
        }

    @app.post("/api/writing/apply")
    async def apply_suggestions(req: ApplySuggestionsRequest):
        """Apply a set of accepted suggestion IDs back into document text."""
        suggestion_set = SuggestionSet()
        accepted_set = set(req.accepted_ids)

        for s_dict in req.suggestions:
            try:
                sug = WritingSuggestion(
                    id=s_dict.get("id"),
                    category=s_dict.get("category", "grammar"),
                    severity=s_dict.get("severity", "info"),
                    original_text=s_dict.get("original_text", ""),
                    suggested_text=s_dict.get("suggested_text", ""),
                    explanation=s_dict.get("explanation", ""),
                    start_offset=s_dict.get("start_offset", 0),
                    end_offset=s_dict.get("end_offset", 0),
                    confidence=s_dict.get("confidence", 0.8),
                )
                if sug.id in accepted_set:
                    sug.accept()
                suggestion_set.add(sug)
            except Exception:
                continue

        updated_text = suggestion_set.apply_to_text(req.text)
        return {"updated_text": updated_text}

    @app.post("/api/clarity")
    async def get_clarity(req: TextAnalysisRequest):
        """Get granular per-sentence clarity metrics."""
        report = clarity_scorer.analyze(req.text)
        return {
            "overall_score": report.overall_clarity_score,
            "avg_fk_grade": report.avg_fk_grade,
            "avg_fog_index": report.avg_fog_index,
            "complexity_distribution": report.complexity_distribution,
            "repetitions": [
                {
                    "type": r.repetition_type,
                    "shared": r.shared_content,
                    "sent_a": r.sentence_a_text,
                    "sent_b": r.sentence_b_text,
                }
                for r in report.repetitions
            ],
        }

    return app


def serve_app(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the AEGIS web server using uvicorn."""
    import uvicorn
    app = create_web_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve_app()
