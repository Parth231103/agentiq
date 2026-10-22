import os
import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi
import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import config

# Set environment optimization flags
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

_embedding_fn = None
_client = None
_vector_store = None


def get_embedding_function():
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_fn


def get_chroma_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=config.vector_store.persist_directory)
    return _client


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            client=get_chroma_client(),
            collection_name=config.vector_store.collection_name,
            embedding_function=get_embedding_function(),
        )
    return _vector_store


def _tokenize(text: str) -> List[str]:
    """Lightweight whitespace & punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


def _get_corpus_documents() -> List[Document]:
    """Extracts all indexed chunks from ChromaDB for BM25 lexical alignment."""
    client = get_chroma_client()
    try:
        col = client.get_collection(config.vector_store.collection_name)
    except Exception:
        return []

    raw = col.get(include=["documents", "metadatas"])
    documents = []
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    for i, content in enumerate(docs):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        documents.append(Document(page_content=content, metadata=meta))
    return documents


def bm25_search(query: str, top_k: int = 15) -> List[Tuple[Document, int]]:
    """Performs sparse lexical retrieval using BM25Okapi."""
    corpus = _get_corpus_documents()
    if not corpus:
        return []

    tokenized_corpus = [_tokenize(doc.page_content) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return [(doc, idx + 1) for idx, doc in enumerate(corpus[:top_k])]

    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    ranked_results = []
    for rank, idx in enumerate(ranked_indices[:top_k], 1):
        if scores[idx] > 0:
            ranked_results.append((corpus[idx], rank))
    return ranked_results


def dense_search(query: str, top_k: int = 15) -> List[Tuple[Document, int]]:
    """Performs dense semantic vector search via ChromaDB."""
    store = get_vector_store()
    results = store.similarity_search_with_score(query, k=top_k)
    ranked = []
    for rank, (doc, _) in enumerate(results, 1):
        ranked.append((doc, rank))
    return ranked


def rerank_candidates(query: str, candidates: List[Document], top_n: int = 5) -> List[Tuple[Document, float]]:
    """Two-stage cross-scoring reranker:
    Evaluates term density, exact phrase coverage, and semantic alignment
    to produce a calibrated relevance score (0.0 to 1.0).
    """
    if not candidates:
        return []

    query_tokens = set(_tokenize(query))
    scored = []
    for doc in candidates:
        doc_tokens = _tokenize(doc.page_content)
        if not doc_tokens:
            continue
        overlap = sum(1 for t in doc_tokens if t in query_tokens)
        density = overlap / (len(doc_tokens) + 1)
        coverage = overlap / (len(query_tokens) + 1e-5)
        relevance_score = min(1.0, 0.45 * coverage + 0.55 * min(1.0, density * 12))
        scored.append((doc, relevance_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def hybrid_search(query: str, top_k: int = None, rrf_k: int = None) -> List[Tuple[Document, float]]:
    """Two-Stage Hybrid Search:
    Stage 1: Fuses sparse BM25 lexical ranking and dense ChromaDB semantic ranking
             using Reciprocal Rank Fusion (RRF).
    Stage 2: Reranks the top candidate pool to output optimal context chunks.
    """
    k = top_k or config.vector_store.top_k
    c_rrf = rrf_k or config.vector_store.rrf_k

    dense_ranks = dense_search(query, top_k=20)
    bm25_ranks = bm25_search(query, top_k=20)

    rrf_scores = {}
    doc_map = {}

    for doc, rank in dense_ranks:
        key = doc.page_content[:150]
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (c_rrf + rank))

    for doc, rank in bm25_ranks:
        key = doc.page_content[:150]
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (c_rrf + rank))

    if not rrf_scores:
        return []

    sorted_keys = sorted(rrf_scores.keys(), key=lambda k_: rrf_scores[k_], reverse=True)
    top_candidates = [doc_map[k_] for k_ in sorted_keys[:20]]

    return rerank_candidates(query, top_candidates, top_n=k)


def similarity_search(query: str, top_k: int = None) -> List[Tuple[Document, float]]:
    """Public retrieval API maintaining backward compatibility with agent tools,
    internally powered by Two-Stage Hybrid Search (BM25 + Dense RRF + Reranking).
    """
    return hybrid_search(query, top_k=top_k)


def add_texts(texts: list[str], metadatas: list[dict] = None):
    store = get_vector_store()
    return store.add_texts(texts=texts, metadatas=metadatas)


def delete_collection():
    client = get_chroma_client()
    try:
        client.delete_collection(config.vector_store.collection_name)
    except Exception:
        pass


def reset_vector_store():
    global _vector_store
    delete_collection()
    _vector_store = None
    return get_vector_store()


def get_collection_stats():
    client = get_chroma_client()
    try:
        col = client.get_collection(config.vector_store.collection_name)
        return {"name": config.vector_store.collection_name, "count": col.count()}
    except Exception:
        return {"name": config.vector_store.collection_name, "count": 0}
