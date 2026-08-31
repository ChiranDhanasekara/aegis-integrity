"""
AEGIS FastAPI REST interface.

Endpoints:
  POST /analyze           -- upload a file and run full analysis
  POST /corpus/add        -- add a document to the comparison corpus
  POST /corpus/build      -- (re)build all indices after adding documents
  GET  /corpus/summary    -- list indexed documents
  POST /compare           -- direct pairwise comparison (no corpus needed)
  POST /batch             -- cross-document essay-mill / classroom analysis
  GET  /health            -- liveness check (no auth required)

All file uploads are handled via multipart/form-data.
Results are returned as JSON (or HTML when ?format=html is appended).

Security:
  Set AEGIS_API_KEY to require an `X-API-Key` header on every route except
  /health. If unset, the API runs without authentication (a warning is
  logged at startup) -- suitable only for a trusted local/private network.
  Uploads are capped at AEGIS_MAX_UPLOAD_MB (default 50MB). Expensive
  analysis routes are limited to AEGIS_MAX_CONCURRENT_JOBS (default 2)
  concurrent requests; additional requests get 503 rather than piling up
  and exhausting memory.

Run with:
    uvicorn aegis.api.app:app --host 127.0.0.1 --port 8000

Binding to 0.0.0.0 (e.g. inside Docker) exposes the API to the network --
set AEGIS_API_KEY when doing so.
"""

from __future__ import annotations
import os
import asyncio
import tempfile
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aegis import __version__ as AEGIS_VERSION
from aegis.core.pipeline import AEGISPipeline, PipelineConfig
from aegis.corpus.indexer import CorpusIndexer
from aegis.report.generator import ReportGenerator
from aegis.writing.rewriter import AcademicRewriter
from aegis.writing.clarity_scorer import ClarityScorer
from aegis.writing.suggestion import SuggestionSet, WritingSuggestion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

INDEX_DIR = os.environ.get("AEGIS_INDEX_DIR", "./aegis_index")
REPORT_DIR = os.environ.get("AEGIS_REPORT_DIR", "./aegis_reports")
DEVICE = os.environ.get("AEGIS_DEVICE", "cpu")
CITATION_EMAIL = os.environ.get("AEGIS_CITATION_EMAIL", "aegis-check@example.com")
API_KEY = os.environ.get("AEGIS_API_KEY")
MAX_UPLOAD_BYTES = int(os.environ.get("AEGIS_MAX_UPLOAD_MB", "50")) * 1024 * 1024
MAX_CONCURRENT_JOBS = int(os.environ.get("AEGIS_MAX_CONCURRENT_JOBS", "2"))

if not API_KEY:
    logger.warning(
        "AEGIS_API_KEY is not set -- the API is running WITHOUT authentication. "
        "Only expose this to a trusted network, or set AEGIS_API_KEY."
    )

_indexer = CorpusIndexer(INDEX_DIR, device=DEVICE)
_pipeline: Optional[AEGISPipeline] = None
_reporter = ReportGenerator(REPORT_DIR)
_job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)


def _get_pipeline() -> AEGISPipeline:
    global _pipeline
    if _pipeline is None:
        cfg = PipelineConfig(
            device=DEVICE,
            citation_email=CITATION_EMAIL,
        )
        _pipeline = AEGISPipeline(config=cfg)
    return _pipeline


def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """FastAPI dependency: no-op if AEGIS_API_KEY is unset (auth disabled);
    otherwise requires a matching X-API-Key header on every protected route."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


@asynccontextmanager
async def _job_slot():
    """Bound the number of concurrent expensive analysis jobs so unbounded
    parallel uploads can't exhaust memory/CPU. Returns 503 when full rather
    than queuing indefinitely."""
    if _job_semaphore.locked():
        raise HTTPException(
            status_code=503,
            detail=f"Server busy: {MAX_CONCURRENT_JOBS} analysis job(s) already running")
    await _job_semaphore.acquire()
    try:
        yield
    finally:
        _job_semaphore.release()


async def _read_upload_capped(file: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    """Read an upload in chunks, rejecting it as soon as it exceeds max_bytes
    instead of buffering an unbounded amount of attacker-controlled data."""
    chunks = []
    total = 0
    chunk_size = 1024 * 1024
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds the {max_bytes // (1024 * 1024)}MB limit")
        chunks.append(chunk)
    return b"".join(chunks)


app = FastAPI(
    title="AEGIS Academic Integrity Checker",
    description=(
        "Open-source plagiarism, AI content, citation integrity, "
        "stylometric, and self-plagiarism analysis for academic submissions."
    ),
    version=AEGIS_VERSION,
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "index_dir": INDEX_DIR}


@app.get("/corpus/summary", dependencies=[Depends(require_api_key)])
def corpus_summary():
    return _indexer.corpus_summary()


@app.post("/corpus/add", dependencies=[Depends(require_api_key)])
async def corpus_add(
    file: UploadFile = File(...),
    label: Optional[str] = Form(None),
):
    """
    Add a document to the comparison corpus.
    Call /corpus/build after adding all documents to update the search indices.
    """
    suffix = Path(file.filename).suffix if file.filename else ".bin"
    content = await _read_upload_capped(file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        assigned_label = _indexer.add_document(tmp_path, label=label)
        return {"status": "added", "label": assigned_label}
    finally:
        os.unlink(tmp_path)


@app.post("/corpus/build", dependencies=[Depends(require_api_key)])
def corpus_build(
    num_perm: int = Query(128, description="MinHash permutations"),
    word_threshold: float = Query(0.25),
    char_threshold: float = Query(0.40),
):
    """Rebuild all search indices from the current corpus."""
    _indexer.build_indices(
        num_perm=num_perm,
        word_threshold=word_threshold,
        char_threshold=char_threshold,
    )
    # Reload pipeline so it picks up the new indices
    global _pipeline
    _pipeline = None
    return {"status": "built", "summary": _indexer.corpus_summary()}


@app.post("/analyze", dependencies=[Depends(require_api_key)])
async def analyze(
    file: UploadFile = File(...),
    format: str = Query("json", description="Output format: json | html"),
    run_ai: bool = Query(True),
    run_citations: bool = Query(True),
    run_semantic: bool = Query(True),
    run_stylometric: bool = Query(True),
    run_self_plagiarism: bool = Query(True),
    prior_works: Optional[str] = Form(
        None,
        description="JSON list of prior-work texts: [[label, text], ...]",
    ),
):
    """
    Run the full AEGIS analysis on an uploaded document.

    Returns JSON by default; use ?format=html for a browser-viewable report.
    """
    import json as _json

    suffix = Path(file.filename).suffix if file.filename else ".bin"
    content = await _read_upload_capped(file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        async with _job_slot():
            cfg = PipelineConfig(
                device=DEVICE,
                citation_email=CITATION_EMAIL,
                run_ai_detector=run_ai,
                run_citation_check=run_citations,
                run_semantic=run_semantic,
                run_stylometric=run_stylometric,
                run_self_plagiarism=run_self_plagiarism,
            )
            pipeline = AEGISPipeline(config=cfg)

            # Load corpus from persisted indices if available
            try:
                ngram_det = _indexer.load_ngram_detector()
                pipeline._ngram = ngram_det
                pipeline._corpus_loaded = True
            except FileNotFoundError:
                pass  # No corpus indexed yet; proceed without

            try:
                sem_det = _indexer.load_semantic_detector()
                pipeline._semantic = sem_det
            except (FileNotFoundError, ImportError):
                pass

            # Load prior works if supplied
            if prior_works and pipeline._self_plag:
                try:
                    works = _json.loads(prior_works)
                    pipeline.load_prior_works(works)
                except Exception as exc:
                    logger.warning("Could not parse prior_works: %s", exc)

            report = pipeline.analyze(tmp_path)

    finally:
        os.unlink(tmp_path)

    if format == "html":
        html_path = _reporter.generate_html(report)
        with open(html_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    return JSONResponse(_reporter._report_to_dict(report))


@app.post("/compare", dependencies=[Depends(require_api_key)])
async def compare_pair(
    file_a: UploadFile = File(..., description="First document"),
    file_b: UploadFile = File(..., description="Second document (e.g. prior work)"),
    label_a: str = Form("document_a"),
    label_b: str = Form("document_b"),
):
    """
    Direct pairwise comparison: detects self-plagiarism between two documents
    without needing a pre-built corpus.
    """
    from aegis.detectors.self_plagiarism import SelfPlagiarismDetector
    from aegis.core.document import DocumentParser

    parser = DocumentParser()

    tmp_paths = []
    try:
        for upload_file in (file_a, file_b):
            suffix = Path(upload_file.filename).suffix if upload_file.filename else ".bin"
            content = await _read_upload_capped(upload_file)
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_paths.append(tmp.name)

        text_a = parser.parse(tmp_paths[0]).full_text
        text_b = parser.parse(tmp_paths[1]).full_text
    finally:
        for p in tmp_paths:
            if os.path.exists(p):
                os.unlink(p)

    detector = SelfPlagiarismDetector(use_sbert=False)
    result = detector.compare_documents(text_a, label_a, text_b, label_b)

    return {
        "label_a": label_a,
        "label_b": label_b,
        "overall_overlap_pct": result.overall_overlap_pct,
        "risk_level": result.risk_level,
        "cope_guidance": result.cope_guidance,
        "flags": result.flags,
        "top_passages": [
            {
                "type": p.overlap_type,
                "char_jaccard": p.char_jaccard,
                "word_jaccard": p.word_jaccard,
                "text_a": p.submission_text[:200],
                "text_b": p.source_text[:200],
            }
            for p in result.recycled_passages[:10]
        ],
    }


@app.post("/batch", dependencies=[Depends(require_api_key)])
async def batch(
    files: list[UploadFile] = File(..., description="At least 2 submissions"),
    run_ai: bool = Query(True, description="Score each document for AI content "
                          "(enables the high-AI-score cluster signal)"),
):
    """
    Cross-document essay-mill / classroom analysis: detects near-duplicate
    submissions, shared section structure, and AI-score clustering across
    a set of documents that would each look fine analyzed individually.
    """
    from dataclasses import asdict
    from aegis.core.document import DocumentParser
    from aegis.detectors.batch_analyzer import BatchAnalyzer

    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 files for batch analysis")

    parser = DocumentParser()
    tmp_paths = []
    doc_names, doc_texts = [], []
    try:
        for f in files:
            suffix = Path(f.filename).suffix if f.filename else ".bin"
            content = await _read_upload_capped(f)
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_paths.append(tmp.name)
            doc_names.append(Path(f.filename).stem if f.filename else Path(tmp_paths[-1]).stem)
            doc_texts.append(parser.parse(tmp_paths[-1]).full_text)
    finally:
        for p in tmp_paths:
            if os.path.exists(p):
                os.unlink(p)

    ai_scores = None
    if run_ai:
        try:
            async with _job_slot():
                from aegis.detectors.ai_detector import AIContentDetector
                det = AIContentDetector(device=DEVICE)
                ai_scores = [det.detect(t).document_ensemble_score for t in doc_texts]
        except ImportError:
            logger.warning("transformers/torch not installed; skipping AI-score clustering")

    result = BatchAnalyzer().analyze(doc_names, doc_texts, ai_scores=ai_scores)
    return asdict(result)


# ---------------------------------------------------------------------------
# Writing Assistant & Web UI Routes (v4.0)
# ---------------------------------------------------------------------------

_rewriter_engine = AcademicRewriter()
_clarity_engine = ClarityScorer()

WEB_STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"
if WEB_STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def get_ui():
    """Serve the interactive AEGIS web UI."""
    index_file = WEB_STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>AEGIS Academic Platform API Running</h1>", status_code=200)


class TextAnalysisPayload(BaseModel):
    text: str


class ApplyPayload(BaseModel):
    text: str
    accepted_ids: list[str]
    suggestions: list[dict]


@app.post("/api/writing/analyze")
def api_analyze_writing(payload: TextAnalysisPayload):
    """Run writing assistant and clarity scoring on plain text."""
    if not payload.text.strip():
        return {"suggestions": [], "clarity": None}

    sug_set = _rewriter_engine.analyze(payload.text)
    clarity = _clarity_engine.analyze(payload.text)

    return {
        "suggestions": [s.to_dict() for s in sug_set.all],
        "summary": sug_set.summary,
        "clarity": {
            "overall_score": clarity.overall_clarity_score,
            "fk_grade": clarity.avg_fk_grade,
            "fog_index": clarity.avg_fog_index,
            "avg_sentence_words": clarity.avg_sentence_words,
            "total_sentences": clarity.total_sentences,
            "coherence": round(
                sum(p.overall_coherence for p in clarity.paragraph_coherence) /
                max(len(clarity.paragraph_coherence), 1), 2
            ) if clarity.paragraph_coherence else 0.8,
        },
    }


@app.post("/api/writing/apply")
def api_apply_suggestions(payload: ApplyPayload):
    """Apply accepted suggestions to document text."""
    sug_set = SuggestionSet()
    accepted_ids = set(payload.accepted_ids)

    for s_dict in payload.suggestions:
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
            if sug.id in accepted_ids:
                sug.accept()
            sug_set.add(sug)
        except Exception:
            continue

    updated = sug_set.apply_to_text(payload.text)
    return {"updated_text": updated}

