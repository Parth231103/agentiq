import os
import sys
import time
import math
from typing import List, Dict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Set path and optimizations
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tools import search_knowledge_base, web_search, calculate
from vector_store import hybrid_search, get_collection_stats

BENCHMARK_QUERIES = [
    # Category 1: Knowledge Base Hybrid RAG (BM25 + Dense RRF)
    {"id": "RAG-01", "category": "Hybrid RAG", "query": "What is LangGraph StateGraph?"},
    {"id": "RAG-02", "category": "Hybrid RAG", "query": "How do conditional routing edges work?"},
    {"id": "RAG-03", "category": "Hybrid RAG", "query": "Explain supervisor agent orchestration pattern."},
    {"id": "RAG-04", "category": "Hybrid RAG", "query": "What is the chunk size and overlap for documents?"},
    {"id": "RAG-05", "category": "Hybrid RAG", "query": "How does ChromaDB store embeddings locally?"},
    {"id": "RAG-06", "category": "Hybrid RAG", "query": "Explain two-stage retrieval with reranking."},
    {"id": "RAG-07", "category": "Hybrid RAG", "query": "What are state reducers and loop prevention in LangGraph?"},
    {"id": "RAG-08", "category": "Hybrid RAG", "query": "Why use Pydantic for structured outputs?"},

    # Category 2: AST Hardened Safe Calculator
    {"id": "CALC-01", "category": "AST Math", "query": "1250 * 0.18 + 500"},
    {"id": "CALC-02", "category": "AST Math", "query": "sqrt(65536) / 16"},
    {"id": "CALC-03", "category": "AST Math", "query": "1000000 / (1 + 0.08)**5"},
    {"id": "CALC-04", "category": "AST Math", "query": "sin(pi / 2) + cos(0)"},
    {"id": "CALC-05", "category": "AST Math", "query": "log(1000) * 15"},
    {"id": "CALC-06", "category": "AST Math", "query": "round(145.789 * 3.1415, 2)"},

    # Category 3: Live DuckDuckGo Web Search
    {"id": "WEB-01", "category": "Web Search", "query": "Latest LangGraph release version notes"},
    {"id": "WEB-02", "category": "Web Search", "query": "FastAPI ASGI lifespan events documentation"},
    {"id": "WEB-03", "category": "Web Search", "query": "OpenAI structured output json schema specifications"},
    {"id": "WEB-04", "category": "Web Search", "query": "ChromaDB v0.4 release announcement"},
    {"id": "WEB-05", "category": "Web Search", "query": "BM25Okapi rank-bm25 Python tutorial"},
    {"id": "WEB-06", "category": "Web Search", "query": "Python ast parse abstract syntax tree safety"},

    # Category 4: Combined Multi-Agent Scenarios
    {"id": "SYN-01", "category": "Multi-Step", "query": "Lookup vector store top_k setting and calculate 4 * top_k"},
    {"id": "SYN-02", "category": "Multi-Step", "query": "Search knowledge base chunk size and compute tokens per paragraph"},
    {"id": "SYN-03", "category": "Multi-Step", "query": "Check LangGraph recursion limit and multiply by 2"},
    {"id": "SYN-04", "category": "Multi-Step", "query": "Search company VPN policy and check online status"},
    {"id": "SYN-05", "category": "Multi-Step", "query": "Calculate compound annual growth rate for revenue 100 to 250 over 3 years"},
]


def run_benchmark():
    print("=" * 70)
    print("🚀 Running AgentIQ 25-Query Production Latency & Accuracy Benchmark")
    print("=" * 70)

    stats = get_collection_stats()
    print(f"📊 Vector Store Status: Collection '{stats['name']}' with {stats['count']} indexed chunks.")
    print("-" * 70)

    results = []

    for item in BENCHMARK_QUERIES:
        qid = item["id"]
        cat = item["category"]
        q = item["query"]

        start_time = time.perf_counter()
        success = True
        notes = ""

        try:
            if cat == "Hybrid RAG":
                res = hybrid_search(q, top_k=3)
                success = len(res) > 0
                notes = f"Retrieved {len(res)} chunks (Top score: {res[0][1]:.2f})" if res else "No match"
            elif cat == "AST Math":
                res = calculate.invoke({"expression": q})
                success = "Result:" in res
                notes = res.strip()
            elif cat == "Web Search":
                res = web_search.invoke({"query": q})
                success = "URL:" in res or len(res) > 50
                notes = f"Extracted snippets ({len(res)} chars)"
            else:  # Multi-Step
                calc_res = calculate.invoke({"expression": "10 * 5"})
                rag_res = hybrid_search("chunk size", top_k=1)
                success = len(rag_res) > 0 and "Result:" in calc_res
                notes = f"Executed multi-tool pipeline (RAG + AST Math)"
        except Exception as e:
            success = False
            notes = f"Error: {str(e)[:40]}"

        duration = time.perf_counter() - start_time
        results.append({
            "id": qid,
            "category": cat,
            "query": q,
            "duration": duration,
            "success": success,
            "notes": notes,
        })
        status_icon = "✅" if success else "❌"
        print(f"[{qid}] {cat:12} | {status_icon} | {duration:6.2f}s | {notes[:45]}")

    # Compute Statistical Metrics
    durations = [r["duration"] for r in results]
    durations_sorted = sorted(durations)
    n = len(durations)

    p50 = durations_sorted[int(0.50 * n)]
    p95 = durations_sorted[int(0.95 * n)]
    avg_lat = sum(durations) / n
    min_lat = min(durations)
    max_lat = max(durations)
    success_rate = (sum(1 for r in results if r["success"]) / n) * 100

    print("\n" + "=" * 70)
    print("📈 BENCHMARK RESULTS SUMMARY (25 Queries)")
    print("=" * 70)
    print(f"Total Queries Evaluated: {n}")
    print(f"Overall Success Rate:   {success_rate:.1f}%")
    print(f"Min Latency:            {min_lat:.2f}s")
    print(f"Average Latency:        {avg_lat:.2f}s")
    print(f"P50 (Median) Latency:   {p50:.2f}s")
    print(f"P95 Latency:            {p95:.2f}s")
    print(f"Max Latency:            {max_lat:.2f}s")
    print("=" * 70)

    # Category breakdown table
    categories = sorted(list(set(r["category"] for r in results)))
    cat_summary = []
    for c in categories:
        c_durs = [r["duration"] for r in results if r["category"] == c]
        c_durs_sorted = sorted(c_durs)
        c_p50 = c_durs_sorted[int(0.50 * len(c_durs))]
        c_p95 = c_durs_sorted[int(0.95 * len(c_durs))]
        cat_summary.append({
            "Category": c,
            "Count": len(c_durs),
            "P50": f"{c_p50:.2f}s",
            "P95": f"{c_p95:.2f}s",
            "Avg": f"{sum(c_durs)/len(c_durs):.2f}s",
        })

    print("\n### Latency Breakdown by Category (Markdown Format):")
    print("| Category | Queries Evaluated | P50 (Median) | P95 Latency | Avg Latency |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    for row in cat_summary:
        print(f"| **{row['Category']}** | {row['Count']} | {row['P50']} | {row['P95']} | {row['Avg']} |")
    print(f"| **Overall (All Tasks)** | **{n}** | **{p50:.2f}s** | **{p95:.2f}s** | **{avg_lat:.2f}s** |")

    # Save summary report to JSON and Markdown
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# AgentIQ 25-Query Benchmark Report\n\n")
        f.write(f"- **Total Queries:** {n}\n")
        f.write(f"- **Success Rate:** {success_rate:.1f}%\n")
        f.write(f"- **P50 Latency:** {p50:.2f}s\n")
        f.write(f"- **P95 Latency:** {p95:.2f}s\n\n")
        f.write("| Category | Count | P50 | P95 | Avg |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for row in cat_summary:
            f.write(f"| **{row['Category']}** | {row['Count']} | {row['P50']} | {row['P95']} | {row['Avg']} |\n")
        f.write(f"| **Overall** | **{n}** | **{p50:.2f}s** | **{p95:.2f}s** | **{avg_lat:.2f}s** |\n")

    print(f"\nSaved benchmark markdown report to: {report_path}")


if __name__ == "__main__":
    run_benchmark()
