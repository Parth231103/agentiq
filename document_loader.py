import os
from pathlib import Path
from typing import List
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from config import config
from vector_store import get_vector_store


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.vector_store.chunk_size,
        chunk_overlap=config.vector_store.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )


def load_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_text(file_path: str) -> List[Document]:
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def load_url(url: str) -> List[Document]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    title = soup.title.string.strip() if soup.title else url

    return [
        Document(
            page_content=text,
            metadata={"source": url, "title": title, "type": "webpage"},
        )
    ]


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = get_text_splitter()
    return splitter.split_documents(docs)


def ingest_documents(docs: List[Document]) -> int:
    chunks = split_documents(docs)
    if not chunks:
        return 0

    store = get_vector_store()
    store.add_documents(chunks)
    return len(chunks)


def ingest_file(file_path: str) -> int:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        docs = load_pdf(str(path))
    elif ext in [".txt", ".md", ".json", ".csv"]:
        docs = load_text(str(path))
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return ingest_documents(docs)


def ingest_url(url: str) -> int:
    docs = load_url(url)
    return ingest_documents(docs)


if __name__ == "__main__":
    sample_dir = Path("./data/uploads")
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_file = sample_dir / "sample_policy.txt"

    content = """Company Remote Work Policy (2026)

Section 1: Working Hours
Employees are expected to be available during core business hours between 10:00 AM and 4:00 PM EST. 
Flexible scheduling outside these core hours is permitted upon manager approval.

Section 2: Equipment and Security
All engineers must use company-issued hardware with disk encryption enabled. 
VPN access is required when accessing production infrastructure or internal databases. 
Public Wi-Fi networks must never be used without an active encrypted VPN tunnel.

Section 3: Communication Protocol
Slack is the primary tool for real-time messaging. 
All project updates, architectural decisions, and agent workflow specifications must be documented in Notion. 
Code reviews must be submitted via pull requests and require at least two approvals.
"""
    sample_file.write_text(content, encoding="utf-8")
    print(f"Created test file: {sample_file}")

    print("--- Ingesting local text file ---")
    num_chunks = ingest_file(str(sample_file))
    print(f"File successfully ingested into ChromaDB. Total chunks created: {num_chunks}")

    from vector_store import similarity_search
    print("\n--- Verifying retrieval from ingested document ---")
    query = "What are the rules regarding VPN and public Wi-Fi?"
    results = similarity_search(query, top_k=1)
    for doc, score in results:
        print(f"Match Score: {score:.4f}")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Snippet:\n{doc.page_content.strip()}")
