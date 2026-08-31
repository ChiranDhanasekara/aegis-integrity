"""
Open-Access Scholarly Source Integration -- AEGIS v4.0.

Queries freely available academic databases to retrieve abstracts and
metadata for similarity comparison against the user's submission.

Sources:
  - OpenAlex (already partially used in citation_network) -- expanded
  - Semantic Scholar (S2) -- search by key phrases, retrieve abstracts
  - CORE -- open-access full texts from institutional repositories
  - arXiv -- search/retrieve abstracts for STEM papers

Privacy model:
  - Document content is NEVER sent to external services
  - Only extracted key phrases (up to 10 per document) are used as queries
  - Results are cached locally to avoid repeated lookups
  - Entire module is opt-in only

This module returns (label, abstract_text) pairs that can be fed into
the existing similarity detectors (ngram, semantic) as a dynamically
constructed corpus.
"""

from __future__ import annotations
import re
import json
import time
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import Counter

logger = logging.getLogger(__name__)

# Rate limit: minimum seconds between API calls to each service
_RATE_LIMITS = {
    "openalex": 0.1,
    "semantic_scholar": 1.0,
    "core": 1.0,
    "arxiv": 3.0,
}


@dataclass
class SourceResult:
    """A single result from an open-access source query."""
    source_api: str          # "openalex" | "semantic_scholar" | "core" | "arxiv"
    title: str
    authors: list[str]
    year: Optional[int]
    abstract: str
    doi: Optional[str]
    url: Optional[str]
    relevance_score: float   # 0.0–1.0 (from the API or computed)


@dataclass
class OpenAccessSearchResult:
    """Aggregated results from all open-access sources."""
    key_phrases_used: list[str]
    results: list[SourceResult] = field(default_factory=list)
    corpus_pairs: list[tuple[str, str]] = field(default_factory=list)
    sources_queried: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    cache_hits: int = 0
    api_calls: int = 0

    @property
    def total_results(self) -> int:
        return len(self.results)


class OpenAccessCorpusBuilder:
    """
    Build a comparison corpus from open-access scholarly sources.

    Extracts key phrases from the submission, queries external APIs,
    and returns (label, text) pairs suitable for NGramDetector.build_index()
    and SemanticDetector.build_index().

    Usage::

        builder = OpenAccessCorpusBuilder(cache_dir="./oa_cache")
        result = builder.search(submission_text, max_results_per_source=10)
        corpus = result.corpus_pairs  # [(label, abstract_text), ...]
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_openalex: bool = True,
        use_semantic_scholar: bool = True,
        use_core: bool = False,       # Requires API key
        use_arxiv: bool = True,
        core_api_key: Optional[str] = None,
    ):
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._use_openalex = use_openalex
        self._use_semantic_scholar = use_semantic_scholar
        self._use_core = use_core and core_api_key
        self._use_arxiv = use_arxiv
        self._core_api_key = core_api_key
        self._last_call: dict[str, float] = {}

    def search(
        self,
        text: str,
        max_results_per_source: int = 10,
        max_key_phrases: int = 5,
    ) -> OpenAccessSearchResult:
        """
        Extract key phrases from text, query open-access sources, and
        return aggregated results.
        """
        key_phrases = self._extract_key_phrases(text, max_key_phrases)
        result = OpenAccessSearchResult(key_phrases_used=key_phrases)

        if not key_phrases:
            return result

        # Query each enabled source
        if self._use_openalex:
            self._query_openalex(key_phrases, max_results_per_source, result)

        if self._use_semantic_scholar:
            self._query_semantic_scholar(
                key_phrases, max_results_per_source, result)

        if self._use_arxiv:
            self._query_arxiv(key_phrases, max_results_per_source, result)

        if self._use_core:
            self._query_core(key_phrases, max_results_per_source, result)

        # Deduplicate by title similarity
        result.results = self._deduplicate(result.results)

        # Build corpus pairs
        result.corpus_pairs = [
            (f"{r.source_api}:{r.title[:60]}", r.abstract)
            for r in result.results
            if r.abstract and len(r.abstract) > 50
        ]

        return result

    # ------------------------------------------------------------------
    # Key phrase extraction (lightweight, no ML)
    # ------------------------------------------------------------------

    def _extract_key_phrases(self, text: str, max_phrases: int) -> list[str]:
        """
        Extract key phrases using TF-based ranking on bigrams/trigrams.
        No ML model needed -- pure frequency analysis.
        """
        # Tokenize
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())

        # Remove common stop words
        stops = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "has", "have", "been",
            "were", "this", "that", "with", "from", "they", "will", "each",
            "make", "like", "than", "them", "then", "into", "some", "also",
            "very", "when", "what", "more", "most", "which", "their", "about",
            "would", "these", "other", "could", "such", "over", "only",
            "using", "used", "based", "paper", "study", "results", "method",
            "however", "proposed", "approach", "show", "shown", "present",
            "section", "figure", "table",
        }
        filtered = [w for w in words if w not in stops and len(w) > 3]

        # Build bigrams
        bigrams = [f"{filtered[i]} {filtered[i+1]}"
                   for i in range(len(filtered) - 1)]

        # Count and rank
        bigram_counts = Counter(bigrams)
        top_bigrams = bigram_counts.most_common(max_phrases * 2)

        # Filter out very common bigrams and take top N
        phrases = []
        seen_words = set()
        for phrase, count in top_bigrams:
            if count < 2:
                continue
            w1, w2 = phrase.split()
            if w1 not in seen_words or w2 not in seen_words:
                phrases.append(phrase)
                seen_words.update([w1, w2])
            if len(phrases) >= max_phrases:
                break

        # If not enough bigrams, fall back to top unigrams
        if len(phrases) < max_phrases:
            unigram_counts = Counter(filtered)
            for word, count in unigram_counts.most_common(max_phrases * 3):
                if count >= 3 and word not in seen_words:
                    phrases.append(word)
                    seen_words.add(word)
                if len(phrases) >= max_phrases:
                    break

        return phrases

    # ------------------------------------------------------------------
    # OpenAlex
    # ------------------------------------------------------------------

    def _query_openalex(self, phrases: list[str], max_results: int,
                        result: OpenAccessSearchResult) -> None:
        """Query OpenAlex Works API for matching abstracts."""
        import requests

        result.sources_queried.append("openalex")
        query = " ".join(phrases[:3])

        cached = self._load_cache("openalex", query)
        if cached is not None:
            for item in cached[:max_results]:
                result.results.append(self._parse_openalex_work(item))
            result.cache_hits += 1
            return

        self._rate_limit("openalex")
        try:
            url = "https://api.openalex.org/works"
            params = {
                "search": query,
                "per_page": min(max_results, 25),
                "select": "title,authorships,publication_year,doi,abstract_inverted_index",
            }
            resp = requests.get(url, params=params, timeout=15)
            result.api_calls += 1

            if resp.status_code == 200:
                data = resp.json()
                works = data.get("results", [])
                self._save_cache("openalex", query, works)
                for work in works[:max_results]:
                    sr = self._parse_openalex_work(work)
                    if sr.abstract:
                        result.results.append(sr)
            else:
                result.errors.append(
                    f"OpenAlex HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as exc:
            result.errors.append(f"OpenAlex error: {exc}")

    def _parse_openalex_work(self, work: dict) -> SourceResult:
        """Parse an OpenAlex work into a SourceResult."""
        # Reconstruct abstract from inverted index
        abstract = ""
        inv_idx = work.get("abstract_inverted_index")
        if inv_idx:
            positions: list[tuple[int, str]] = []
            for word, pos_list in inv_idx.items():
                for pos in pos_list:
                    positions.append((pos, word))
            positions.sort()
            abstract = " ".join(w for _, w in positions)

        authors = []
        for authorship in work.get("authorships", [])[:5]:
            author = authorship.get("author", {})
            name = author.get("display_name", "")
            if name:
                authors.append(name)

        return SourceResult(
            source_api="openalex",
            title=work.get("title", ""),
            authors=authors,
            year=work.get("publication_year"),
            abstract=abstract,
            doi=work.get("doi"),
            url=work.get("doi"),
            relevance_score=0.5,
        )

    # ------------------------------------------------------------------
    # Semantic Scholar
    # ------------------------------------------------------------------

    def _query_semantic_scholar(self, phrases: list[str], max_results: int,
                                result: OpenAccessSearchResult) -> None:
        """Query Semantic Scholar API for matching papers."""
        import requests

        result.sources_queried.append("semantic_scholar")
        query = " ".join(phrases[:3])

        cached = self._load_cache("semantic_scholar", query)
        if cached is not None:
            for item in cached[:max_results]:
                result.results.append(self._parse_s2_paper(item))
            result.cache_hits += 1
            return

        self._rate_limit("semantic_scholar")
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": min(max_results, 20),
                "fields": "title,authors,year,abstract,externalIds,url",
            }
            resp = requests.get(url, params=params, timeout=15)
            result.api_calls += 1

            if resp.status_code == 200:
                data = resp.json()
                papers = data.get("data", [])
                self._save_cache("semantic_scholar", query, papers)
                for paper in papers[:max_results]:
                    sr = self._parse_s2_paper(paper)
                    if sr.abstract:
                        result.results.append(sr)
            else:
                result.errors.append(
                    f"Semantic Scholar HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as exc:
            result.errors.append(f"Semantic Scholar error: {exc}")

    def _parse_s2_paper(self, paper: dict) -> SourceResult:
        """Parse a Semantic Scholar paper into a SourceResult."""
        authors = [a.get("name", "") for a in paper.get("authors", [])[:5]]
        ext_ids = paper.get("externalIds", {}) or {}
        doi = ext_ids.get("DOI")

        return SourceResult(
            source_api="semantic_scholar",
            title=paper.get("title", ""),
            authors=authors,
            year=paper.get("year"),
            abstract=paper.get("abstract") or "",
            doi=doi,
            url=paper.get("url"),
            relevance_score=0.5,
        )

    # ------------------------------------------------------------------
    # arXiv
    # ------------------------------------------------------------------

    def _query_arxiv(self, phrases: list[str], max_results: int,
                     result: OpenAccessSearchResult) -> None:
        """Query arXiv API for matching papers."""
        import requests

        result.sources_queried.append("arxiv")
        query = "+AND+".join(f"all:{p.replace(' ', '+')}" for p in phrases[:3])

        cached = self._load_cache("arxiv", query)
        if cached is not None:
            for item in cached[:max_results]:
                result.results.append(item)
            result.cache_hits += 1
            return

        self._rate_limit("arxiv")
        try:
            url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": query,
                "start": 0,
                "max_results": min(max_results, 20),
            }
            resp = requests.get(url, params=params, timeout=15)
            result.api_calls += 1

            if resp.status_code == 200:
                papers = self._parse_arxiv_xml(resp.text)
                # Cache as dicts for serialization
                cache_data = [
                    {"title": p.title, "authors": p.authors, "year": p.year,
                     "abstract": p.abstract, "doi": p.doi, "url": p.url}
                    for p in papers
                ]
                self._save_cache("arxiv", query, cache_data)
                for paper in papers[:max_results]:
                    if paper.abstract:
                        result.results.append(paper)
            else:
                result.errors.append(
                    f"arXiv HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as exc:
            result.errors.append(f"arXiv error: {exc}")

    def _parse_arxiv_xml(self, xml_text: str) -> list[SourceResult]:
        """Parse arXiv Atom XML response."""
        import xml.etree.ElementTree as ET
        results = []
        try:
            root = ET.fromstring(xml_text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                title = title_el.text.strip() if title_el is not None and title_el.text else ""
                abstract = summary_el.text.strip() if summary_el is not None and summary_el.text else ""

                authors = []
                for author_el in entry.findall("atom:author", ns):
                    name_el = author_el.find("atom:name", ns)
                    if name_el is not None and name_el.text:
                        authors.append(name_el.text.strip())

                # Extract year from published date
                pub_el = entry.find("atom:published", ns)
                year = None
                if pub_el is not None and pub_el.text:
                    year_m = re.search(r"(\d{4})", pub_el.text)
                    if year_m:
                        year = int(year_m.group(1))

                # Extract DOI from links
                doi = None
                for link in entry.findall("atom:link", ns):
                    href = link.get("href", "")
                    if "doi.org" in href:
                        doi = href

                url_el = entry.find("atom:id", ns)
                url = url_el.text.strip() if url_el is not None and url_el.text else None

                results.append(SourceResult(
                    source_api="arxiv",
                    title=title,
                    authors=authors[:5],
                    year=year,
                    abstract=abstract,
                    doi=doi,
                    url=url,
                    relevance_score=0.5,
                ))
        except ET.ParseError as exc:
            logger.warning("arXiv XML parse error: %s", exc)

        return results

    # ------------------------------------------------------------------
    # CORE (requires API key)
    # ------------------------------------------------------------------

    def _query_core(self, phrases: list[str], max_results: int,
                    result: OpenAccessSearchResult) -> None:
        """Query CORE API for open-access full texts."""
        import requests

        result.sources_queried.append("core")
        query = " ".join(phrases[:3])

        cached = self._load_cache("core", query)
        if cached is not None:
            for item in cached[:max_results]:
                result.results.append(self._parse_core_result(item))
            result.cache_hits += 1
            return

        self._rate_limit("core")
        try:
            url = "https://api.core.ac.uk/v3/search/works"
            headers = {"Authorization": f"Bearer {self._core_api_key}"}
            params = {"q": query, "limit": min(max_results, 20)}
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            result.api_calls += 1

            if resp.status_code == 200:
                data = resp.json()
                works = data.get("results", [])
                self._save_cache("core", query, works)
                for work in works[:max_results]:
                    sr = self._parse_core_result(work)
                    if sr.abstract:
                        result.results.append(sr)
            else:
                result.errors.append(
                    f"CORE HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as exc:
            result.errors.append(f"CORE error: {exc}")

    def _parse_core_result(self, work: dict) -> SourceResult:
        """Parse a CORE API result."""
        return SourceResult(
            source_api="core",
            title=work.get("title", ""),
            authors=[a.get("name", "") for a in work.get("authors", [])[:5]],
            year=work.get("yearPublished"),
            abstract=work.get("abstract") or "",
            doi=work.get("doi"),
            url=work.get("downloadUrl"),
            relevance_score=0.5,
        )

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self, results: list[SourceResult]) -> list[SourceResult]:
        """Remove duplicate papers (same title, case-insensitive)."""
        seen_titles: set[str] = set()
        unique = []
        for r in results:
            norm_title = re.sub(r"\s+", " ", r.title.lower().strip())
            if norm_title and norm_title not in seen_titles:
                seen_titles.add(norm_title)
                unique.append(r)
        return unique

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _cache_key(self, source: str, query: str) -> str:
        h = hashlib.md5(f"{source}:{query}".encode()).hexdigest()[:16]
        return f"{source}_{h}.json"

    def _load_cache(self, source: str, query: str) -> Optional[list]:
        if not self._cache_dir:
            return None
        path = self._cache_dir / self._cache_key(source, query)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Cache expiry: 7 days
                if time.time() - data.get("timestamp", 0) < 7 * 86400:
                    return data.get("results", [])
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    def _save_cache(self, source: str, query: str, results: list) -> None:
        if not self._cache_dir:
            return
        path = self._cache_dir / self._cache_key(source, query)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"timestamp": time.time(), "results": results},
                          f, ensure_ascii=False, default=str)
        except Exception as exc:
            logger.debug("Cache write failed: %s", exc)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _rate_limit(self, source: str) -> None:
        min_interval = _RATE_LIMITS.get(source, 1.0)
        last = self._last_call.get(source, 0)
        elapsed = time.time() - last
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call[source] = time.time()
