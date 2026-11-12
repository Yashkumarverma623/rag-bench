"""FastAPI serving layer and interactive query inspector."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, cast

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag_bench.data import load_beir_dataset
from rag_bench.retrievers.base import format_doc_text
from rag_bench.retrievers.bm25 import BM25Retriever, tokenize
from rag_bench.retrievers.dense_exact import DenseExactRetriever
from rag_bench.retrievers.dense_hnsw import DenseHNSWRetriever
from rag_bench.retrievers.hybrid_rrf import rrf_fuse
from rag_bench.retrievers.hybrid_score_fusion import score_fuse
from rag_bench.retrievers.reranker import CrossEncoderReranker

app = FastAPI(title="rag-bench Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dataset and retriever cache
DATASET_CACHE: Dict[str, Any] = {}
RETRIEVER_CACHE: Dict[str, Any] = {}


def get_or_load_dataset(dataset_name: str) -> Dict[str, Any]:
    dataset_name = dataset_name.lower().strip()
    if dataset_name not in DATASET_CACHE:
        dataset = load_beir_dataset(dataset_name)
        # Pre-build retrievers
        bm25 = BM25Retriever()
        bm25.index(dataset.corpus)

        dense = DenseExactRetriever()
        dense.index(dataset.corpus)

        hnsw = DenseHNSWRetriever(exact_retriever=dense)
        hnsw.index(dataset.corpus)

        reranker = CrossEncoderReranker()

        DATASET_CACHE[dataset_name] = {
            "dataset": dataset,
            "bm25": bm25,
            "dense": dense,
            "hnsw": hnsw,
            "reranker": reranker,
        }
    return cast(Dict[str, Any], DATASET_CACHE[dataset_name])


class QueryRequest(BaseModel):
    query: str
    dataset: str = "scifact"
    mode: str = "hybrid_rrf"
    top_k: int = 10


class QueryResultItem(BaseModel):
    rank: int
    doc_id: str
    score: float
    snippet: str
    per_channel: Dict[str, Any]


class QueryResponse(BaseModel):
    query: str
    dataset: str
    mode: str
    top_k: int
    results: List[QueryResultItem]
    latencies_ms: Dict[str, float]


@app.post("/query", response_model=QueryResponse)
def execute_query(req: QueryRequest) -> QueryResponse:
    """Execute retrieval query and return ranked documents, channel breakdowns, and stage latencies."""
    if req.dataset not in ["scifact", "nfcorpus"]:
        raise HTTPException(status_code=400, detail=f"Unsupported dataset: {req.dataset}")

    cache = get_or_load_dataset(req.dataset)
    dataset = cache["dataset"]
    bm25 = cache["bm25"]
    dense = cache["dense"]
    hnsw = cache["hnsw"]
    reranker = cache["reranker"]

    stage_timings: Dict[str, float] = {}
    t_start = time.perf_counter()

    # 1. Tokenize
    t0 = time.perf_counter()
    _ = tokenize(req.query)
    stage_timings["tokenize_ms"] = (time.perf_counter() - t0) * 1000.0

    # 2. BM25 Search
    t0 = time.perf_counter()
    bm25_res = bm25.search(req.query, top_k=max(req.top_k * 2, 50))
    stage_timings["bm25_ms"] = (time.perf_counter() - t0) * 1000.0

    # 3. Dense Query Embed & Search
    t0 = time.perf_counter()
    dense_res = dense.search(req.query, top_k=max(req.top_k * 2, 50))
    stage_timings["dense_exact_ms"] = (time.perf_counter() - t0) * 1000.0

    # Channel lookup maps
    bm25_ranks = {doc: r for r, (doc, _) in enumerate(bm25_res, start=1)}
    bm25_scores = {doc: s for doc, s in bm25_res}
    dense_ranks = {doc: r for r, (doc, _) in enumerate(dense_res, start=1)}
    dense_scores = {doc: s for doc, s in dense_res}

    # Execute specified mode
    final_ranked: List[tuple[str, float]] = []

    if req.mode == "bm25":
        final_ranked = bm25_res[:req.top_k]
    elif req.mode == "dense_exact":
        final_ranked = dense_res[:req.top_k]
    elif req.mode == "dense_hnsw":
        t0 = time.perf_counter()
        final_ranked = hnsw.search(req.query, top_k=req.top_k)
        stage_timings["hnsw_ms"] = (time.perf_counter() - t0) * 1000.0
    elif req.mode == "hybrid_rrf":
        t0 = time.perf_counter()
        final_ranked = rrf_fuse({"bm25": bm25_res, "dense": dense_res}, k=60, top_k=req.top_k)
        stage_timings["fusion_ms"] = (time.perf_counter() - t0) * 1000.0
    elif req.mode == "hybrid_score_fusion":
        t0 = time.perf_counter()
        final_ranked = score_fuse(bm25_res, dense_res, alpha=0.5, top_k=req.top_k)
        stage_timings["fusion_ms"] = (time.perf_counter() - t0) * 1000.0
    elif "rerank" in req.mode:
        t0 = time.perf_counter()
        fused = rrf_fuse({"bm25": bm25_res, "dense": dense_res}, k=60, top_k=50)
        final_ranked = reranker.rerank(req.query, fused, dataset.corpus, top_n=20, top_k=req.top_k)
        stage_timings["rerank_ms"] = (time.perf_counter() - t0) * 1000.0
    else:
        final_ranked = rrf_fuse({"bm25": bm25_res, "dense": dense_res}, k=60, top_k=req.top_k)

    stage_timings["total_ms"] = (time.perf_counter() - t_start) * 1000.0

    items: List[QueryResultItem] = []
    for rank, (doc_id, score) in enumerate(final_ranked[:req.top_k], start=1):
        doc_obj = dataset.corpus.get(doc_id, {})
        full_text = format_doc_text(doc_obj)
        snippet = full_text[:280] + ("..." if len(full_text) > 280 else "")

        per_ch = {
            "bm25_rank": bm25_ranks.get(doc_id, ">50"),
            "bm25_score": bm25_scores.get(doc_id, 0.0),
            "dense_rank": dense_ranks.get(doc_id, ">50"),
            "dense_score": dense_scores.get(doc_id, 0.0),
        }

        items.append(
            QueryResultItem(
                rank=rank,
                doc_id=doc_id,
                score=float(score),
                snippet=snippet,
                per_channel=per_ch,
            )
        )

    return QueryResponse(
        query=req.query,
        dataset=req.dataset,
        mode=req.mode,
        top_k=len(items),
        results=items,
        latencies_ms=stage_timings,
    )


@app.get("/results")
def get_results() -> Dict[str, Any]:
    """Return stored benchmark results across all completed runs."""
    results_dir = Path("results")
    if not results_dir.exists():
        return {"runs": []}

    runs: List[Dict[str, Any]] = []
    for d in sorted(results_dir.iterdir(), reverse=True):
        if d.is_dir() and (d / "summary_metrics.json").exists():
            try:
                with open(d / "manifest.json", "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                with open(d / "summary_metrics.json", "r", encoding="utf-8") as f:
                    summary = json.load(f)
                with open(d / "latency_profiles.json", "r", encoding="utf-8") as f:
                    latency = json.load(f)
                runs.append({
                    "run_id": d.name,
                    "dataset": manifest.get("dataset"),
                    "git_sha": manifest.get("git_sha"),
                    "summary_metrics": summary,
                    "system_latencies": latency.get("systems", {}),
                })
            except Exception:
                continue

    return {"runs": runs}


# Static web interface
STATIC_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>rag-bench: Retrieval Pipeline Benchmark & Inspector</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --text: #e6edf3;
      --text-muted: #8b949e;
      --primary: #58a6ff;
      --primary-hover: #79c0ff;
      --accent: #238636;
      --accent-hover: #2ea043;
      --tag-bg: #21262d;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }
    h1 { font-size: 24px; font-weight: 700; color: #fff; margin-bottom: 6px; }
    p.subtitle { color: var(--text-muted); font-size: 14px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
    }
    .card h2 { font-size: 16px; font-weight: 600; margin-bottom: 14px; color: var(--primary); }
    .form-group { margin-bottom: 14px; }
    label { display: block; font-size: 12px; font-weight: 600; color: var(--text-muted); margin-bottom: 6px; text-transform: uppercase; }
    input[type="text"], select {
      width: 100%;
      padding: 10px 12px;
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      color: #fff;
      font-family: inherit;
      font-size: 14px;
    }
    input[type="text"]:focus, select:focus {
      outline: none;
      border-color: var(--primary);
    }
    button.btn {
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 10px 18px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 14px;
      cursor: pointer;
      transition: background 0.15s ease;
    }
    button.btn:hover { background: var(--accent-hover); }
    .results-pane { margin-top: 16px; }
    .result-item {
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 14px;
      margin-bottom: 12px;
    }
    .result-header {
      display: flex;
      justify-content: space-between;
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      margin-bottom: 6px;
    }
    .badge {
      background: var(--tag-bg);
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 12px;
      border: 1px solid var(--border);
    }
    .channels {
      display: flex;
      gap: 12px;
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 8px;
      font-family: 'JetBrains Mono', monospace;
    }
    .timing-bar {
      display: flex;
      gap: 16px;
      background: #0d1117;
      border: 1px solid var(--border);
      padding: 10px 14px;
      border-radius: 6px;
      font-size: 12px;
      font-family: 'JetBrains Mono', monospace;
      margin-bottom: 16px;
    }
    table.data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    table.data-table th, table.data-table td {
      border: 1px solid var(--border);
      padding: 8px 10px;
      text-align: left;
    }
    table.data-table th { background: #0d1117; color: var(--text-muted); font-weight: 600; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>rag-bench: Retrieval Inspector & Results</h1>
      <p class="subtitle">Multi-system IR benchmark (BM25, Dense, Hybrid RRF, Reranker) with stage latency profiling</p>
    </header>

    <div class="grid">
      <!-- Query Inspector Panel -->
      <div class="card">
        <h2>Interactive Query Inspector</h2>
        <div class="form-group">
          <label for="datasetSelect">Dataset</label>
          <select id="datasetSelect">
            <option value="scifact">BEIR SciFact</option>
            <option value="nfcorpus">BEIR NFCorpus</option>
          </select>
        </div>
        <div class="form-group">
          <label for="modeSelect">Retrieval Mode</label>
          <select id="modeSelect">
            <option value="hybrid_rrf">Hybrid RRF (BM25 + Dense)</option>
            <option value="bm25">BM25 (Sparse CSR)</option>
            <option value="dense_exact">Dense Exact (bge-small Cosine)</option>
            <option value="dense_hnsw">Dense HNSW (Faiss)</option>
            <option value="hybrid_score_fusion">Hybrid Score Fusion (Min-Max Convex)</option>
            <option value="rerank">Cross-Encoder Reranker</option>
          </select>
        </div>
        <div class="form-group">
          <label for="queryInput">Query Text</label>
          <input type="text" id="queryInput" placeholder="e.g. MicroRNA regulation in lung cancer" value="MicroRNA regulation in lung cancer">
        </div>
        <button id="searchBtn" class="btn" onclick="runSearch()">Execute Query</button>

        <div id="timingPane" style="display:none; margin-top:16px;">
          <div class="timing-bar" id="timingContent"></div>
        </div>

        <div class="results-pane" id="resultsPane"></div>
      </div>

      <!-- Results Overview Panel -->
      <div class="card">
        <h2>Stored Benchmark Runs</h2>
        <div id="runsContainer">
          <p style="color:var(--text-muted); font-size:13px;">Loading stored results from /results...</p>
        </div>
      </div>
    </div>
  </div>

  <script>
    async function runSearch() {
      const btn = document.getElementById('searchBtn');
      const query = document.getElementById('queryInput').value.trim();
      const dataset = document.getElementById('datasetSelect').value;
      const mode = document.getElementById('modeSelect').value;
      const pane = document.getElementById('resultsPane');
      const timingPane = document.getElementById('timingPane');
      const timingContent = document.getElementById('timingContent');

      if (!query) return;
      btn.disabled = true;
      btn.innerText = "Querying...";
      pane.innerHTML = "<p style='color:#8b949e; font-size:13px;'>Retrieving documents...</p>";

      try {
        const res = await fetch('/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, dataset, mode, top_k: 10 })
        });
        const data = await res.json();

        // Show stage timings
        const t = data.latencies_ms;
        timingContent.innerHTML = `
          <span>Total: <b>${t.total_ms?.toFixed(1) || 0}ms</b></span>
          <span>BM25: <b>${t.bm25_ms?.toFixed(1) || 0}ms</b></span>
          <span>Dense: <b>${t.dense_exact_ms?.toFixed(1) || 0}ms</b></span>
          ${t.fusion_ms ? `<span>Fusion: <b>${t.fusion_ms.toFixed(1)}ms</b></span>` : ''}
          ${t.rerank_ms ? `<span>Rerank: <b>${t.rerank_ms.toFixed(1)}ms</b></span>` : ''}
        `;
        timingPane.style.display = 'block';

        // Render ranked results
        pane.innerHTML = data.results.map(r => `
          <div class="result-item">
            <div class="result-header">
              <span>#${r.rank} &bull; <b>${r.doc_id}</b></span>
              <span class="badge">score: ${r.score.toFixed(4)}</span>
            </div>
            <div style="font-size:13px; color:#c9d1d9;">${r.snippet}</div>
            <div class="channels">
              <span>BM25 Rank: ${r.per_channel.bm25_rank} (${r.per_channel.bm25_score.toFixed(2)})</span>
              <span>Dense Rank: ${r.per_channel.dense_rank} (${r.per_channel.dense_score.toFixed(2)})</span>
            </div>
          </div>
        `).join('');
      } catch (err) {
        pane.innerHTML = `<p style='color:#f85149;'>Error: ${err.message}</p>`;
      } finally {
        btn.disabled = false;
        btn.innerText = "Execute Query";
      }
    }

    async function loadStoredResults() {
      const container = document.getElementById('runsContainer');
      try {
        const res = await fetch('/results');
        const data = await res.json();
        if (!data.runs || data.runs.length === 0) {
          container.innerHTML = "<p style='color:#8b949e; font-size:13px;'>No completed benchmark runs found in results/.</p>";
          return;
        }

        let html = "";
        for (const run of data.runs) {
          html += `<div style="margin-bottom:18px;">
            <div style="font-weight:600; font-size:14px; color:#fff; margin-bottom:4px;">${run.dataset} (${run.run_id})</div>
            <table class="data-table">
              <thead><tr><th>System</th><th>nDCG@10</th><th>R@100</th><th>p95 (ms)</th></tr></thead>
              <tbody>`;
          for (const [sys, m] of Object.entries(run.summary_metrics)) {
            const lat = run.system_latencies[sys]?.p95_ms || 0;
            html += `<tr><td><code>${sys}</code></td><td>${m['ndcg@10'].toFixed(4)}</td><td>${m['recall@100'].toFixed(4)}</td><td>${lat.toFixed(1)}</td></tr>`;
          }
          html += `</tbody></table></div>`;
        }
        container.innerHTML = html;
      } catch (e) {
        container.innerHTML = "<p style='color:#f85149; font-size:13px;'>Failed to load results.</p>";
      }
    }

    loadStoredResults();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index_page() -> str:
    """Serve single static query inspector page."""
    return STATIC_HTML
