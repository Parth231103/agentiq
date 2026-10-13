import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import config


def get_embedding_function():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


embedding_fn = get_embedding_function()


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=config.vector_store.collection_name,
        embedding_function=embedding_fn,
        persist_directory=config.vector_store.persist_directory,
    )


def similarity_search(query: str, top_k: int = None):
    k = top_k or config.vector_store.top_k
    store = get_vector_store()
    results = store.similarity_search_with_score(query, k=k)
    return [(doc, 1 - score) for doc, score in results]


def add_texts(texts: list[str], metadatas: list[dict] = None):
    store = get_vector_store()
    return store.add_texts(texts=texts, metadatas=metadatas)


def delete_collection():
    client = chromadb.PersistentClient(path=config.vector_store.persist_directory)
    try:
        client.delete_collection(config.vector_store.collection_name)
    except Exception:
        pass


def reset_vector_store():
    delete_collection()
    return get_vector_store()


def get_collection_stats():
    store = get_vector_store()
    collection = store._collection
    return {
        "name": config.vector_store.collection_name,
        "count": collection.count(),
    }


vector_store = get_vector_store()
