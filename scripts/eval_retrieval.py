import os
import sys
from typing import List, Dict, Any, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from vector_store import dense_search, bm25_search, hybrid_search, _get_corpus_documents

# 20 ground truth evaluation queries with expected target keywords or content substrings
EVAL_SET = [
    {
        "query": "Why do we divide by sqrt(dk) in self-attention formula?",
        "target_keyword": "sqrt(dk)"
    },
    {
        "query": "How does KV caching optimize multi-token generation?",
        "target_keyword": "KV"
    },
    {
        "query": "What is the difference between top-k and top-p nucleus sampling?",
        "target_keyword": "Top-k"
    },
    {
        "query": "How does Sparse Mixture of Experts MoE routing work?",
        "target_keyword": "Mixture-of-Experts"
    },
    {
        "query": "Why do LLMs make arithmetic errors due to tokenization?",
        "target_keyword": "math must always be offloaded"
    },
    {
        "query": "How does Reciprocal Rank Fusion RRF merge search results?",
        "target_keyword": "Rank Fusion (RRF)"
    },
    {
        "query": "Why can BM25 scores and cosine vector similarity not be added directly?",
        "target_keyword": "cannot be added directly"
    },
    {
        "query": "What is the computational difference between Cross-Encoder and Bi-Encoder?",
        "target_keyword": "Self-Attention runs across every query token"
    },
    {
        "query": "Explain the two-stage retrieval architecture for RAG systems.",
        "target_keyword": "Two-Stage Retrieval"
    },
    {
        "query": "How does the Parent-Document retriever handle chunk sizing?",
        "target_keyword": "Parent-Documen"
    },
    {
        "query": "How do IVF-Flat index and Voronoi cells work in vector search?",
        "target_keyword": "Voronoi"
    },
    {
        "query": "What is the lost in the middle effect in long context prompts?",
        "target_keyword": "very beginning and very end"
    },
    {
        "query": "How does LangGraph model cyclic workflows with StateGraph?",
        "target_keyword": "cyclic graphs with explicitState"
    },
    {
        "query": "How do you prevent an autonomous agent from getting stuck in an infinite loop?",
        "target_keyword": "stuck in an in"
    },
    {
        "query": "How do token and cost budget circuit breakers protect agent execution?",
        "target_keyword": "Budget Circuit"
    },
    {
        "query": "Why should tool calls be validated with Pydantic schemas?",
        "target_keyword": "Pydantic"
    },
    {
        "query": "When should an enterprise choose RAG over Fine-Tuning?",
        "target_keyword": "Fine-Tuning vs. RAG"
    },
    {
        "query": "How do you evaluate RAG systems automatically at scale without manual review?",
        "target_keyword": "automated"
    },
    {
        "query": "What is Time To First Token TTFT and how do you reduce latency?",
        "target_keyword": "Time-To-First-Token"
    },
    {
        "query": "How do you prevent arbitrary code execution RCE when an agent calculates math?",
        "target_keyword": "Code Execution (RCE)"
    }
]


def evaluate_retrieval():
    print("=" * 65)
    print("AgentIQ Retrieval Quality Benchmark: Dense vs. Hybrid vs. Rerank")
    print(f"Total Evaluation Queries: {len(EVAL_SET)}")
    print("=" * 65)

    dense_hits_at_1 = 0
    dense_hits_at_5 = 0
    dense_mrr = 0.0

    hybrid_hits_at_1 = 0
    hybrid_hits_at_5 = 0
    hybrid_mrr = 0.0

    rerank_hits_at_1 = 0
    rerank_hits_at_5 = 0
    rerank_mrr = 0.0

    for item in EVAL_SET:
        q = item["query"]
        target = item["target_keyword"].lower()

        # 1. Dense Only (ChromaDB)
        dense_results = dense_search(q, top_k=5)
        dense_rank = None
        for rank, (doc, _) in enumerate(dense_results, 1):
            if target in doc.page_content.lower():
                dense_rank = rank
                break
        if dense_rank:
            if dense_rank == 1:
                dense_hits_at_1 += 1
            if dense_rank <= 5:
                dense_hits_at_5 += 1
            dense_mrr += 1.0 / dense_rank

        # 2. Hybrid RRF (No cross-scoring rerank)
        # Using top 5 from dense + bm25 RRF directly
        dense_pool = dense_search(q, top_k=15)
        bm25_pool = bm25_search(q, top_k=15)
        rrf = {}
        d_map = {}
        for d, r in dense_pool:
            k = d.page_content[:150]
            d_map[k] = d
            rrf[k] = rrf.get(k, 0.0) + (1.0 / (60 + r))
        for d, r in bm25_pool:
            k = d.page_content[:150]
            d_map[k] = d
            rrf[k] = rrf.get(k, 0.0) + (1.0 / (60 + r))
        sorted_keys = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)
        hybrid_results = [d_map[k] for k in sorted_keys[:5]]

        hybrid_rank = None
        for rank, doc in enumerate(hybrid_results, 1):
            if target in doc.page_content.lower():
                hybrid_rank = rank
                break
        if hybrid_rank:
            if hybrid_rank == 1:
                hybrid_hits_at_1 += 1
            if hybrid_rank <= 5:
                hybrid_hits_at_5 += 1
            hybrid_mrr += 1.0 / hybrid_rank

        # 3. Two-Stage: Hybrid + Cross-Scoring Reranker
        rerank_results = hybrid_search(q, top_k=5)
        rerank_rank = None
        for rank, (doc, _) in enumerate(rerank_results, 1):
            if target in doc.page_content.lower():
                rerank_rank = rank
                break
        if rerank_rank:
            if rerank_rank == 1:
                rerank_hits_at_1 += 1
            if rerank_rank <= 5:
                rerank_hits_at_5 += 1
            rerank_mrr += 1.0 / rerank_rank

    n = len(EVAL_SET)
    print("\nRESULTS SUMMARY:")
    print("-" * 65)
    print(f"{'Method':<25} | {'Hit@1':<10} | {'Hit@5':<10} | {'MRR':<10}")
    print("-" * 65)
    print(f"{'Dense Only (ChromaDB)':<25} | {dense_hits_at_1/n*100:5.1f}%     | {dense_hits_at_5/n*100:5.1f}%     | {dense_mrr/n:5.3f}")
    print(f"{'Hybrid (BM25 + Dense RRF)':<25} | {hybrid_hits_at_1/n*100:5.1f}%     | {hybrid_hits_at_5/n*100:5.1f}%     | {hybrid_mrr/n:5.3f}")
    print(f"{'Two-Stage (+ Reranker)':<25} | {rerank_hits_at_1/n*100:5.1f}%     | {rerank_hits_at_5/n*100:5.1f}%     | {rerank_mrr/n:5.3f}")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_retrieval()
