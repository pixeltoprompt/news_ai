"""Flask API — thin HTTP layer over the ingestion, RAG, and generation modules."""

from __future__ import annotations

from flask import Flask, jsonify, request

from .ingestion import ingest, load_articles
from .rag_chain import answer_query
from .summarization import generate_presenter_script, summarize_article

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/ingest")
def run_ingestion():
    report = ingest()
    return jsonify(report.model_dump())


@app.post("/query")
def query():
    """RAG endpoint. Body: {"query": "...", "top_k": 4}"""
    payload = request.get_json(silent=True) or {}
    q = payload.get("query")
    if not q:
        return jsonify({"error": "query is required"}), 400

    result = answer_query(q, top_k=payload.get("top_k", 4))
    return jsonify(result.model_dump(mode="json"))


@app.post("/summarize/<article_id>")
def summarize(article_id: str):
    articles = {a.article_id: a for a in load_articles()}
    article = articles.get(article_id)
    if not article:
        return jsonify({"error": f"Unknown article_id: {article_id}"}), 404

    result = summarize_article(article)
    return jsonify(result.model_dump())


@app.post("/presenter-script/<article_id>")
def presenter_script(article_id: str):
    payload = request.get_json(silent=True) or {}
    articles = {a.article_id: a for a in load_articles()}
    article = articles.get(article_id)
    if not article:
        return jsonify({"error": f"Unknown article_id: {article_id}"}), 404

    result = generate_presenter_script(article, target_duration_seconds=payload.get("target_duration_seconds", 30))
    return jsonify(result.model_dump())


if __name__ == "__main__":
    app.run(debug=True, port=5001)
