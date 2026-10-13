import os
from pathlib import Path
import streamlit as st
from graph import stream_workflow
from document_loader import ingest_file, ingest_url, ingest_documents
from vector_store import get_collection_stats, get_vector_store
from langchain_core.documents import Document
from config import config

st.set_page_config(page_title="AgentIQ", layout="wide")


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def get_indexed_sources():
    try:
        store = get_vector_store()
        data = store._collection.get()
        sources = set(m.get("source", "") for m in data.get("metadatas", []) if m)
        return [s for s in sources if s]
    except Exception:
        return []


init_session_state()

with st.sidebar:
    st.title("AgentIQ Settings")
    st.caption("Multi-Agent Research Assistant")

    st.divider()
    st.subheader("Knowledge Base")
    stats = get_collection_stats()
    st.metric(label="Indexed Chunks in Vector DB", value=stats["count"])

    sources = get_indexed_sources()
    if sources:
        with st.expander(f"Active Sources ({len(sources)})", expanded=False):
            for s in sources:
                st.write(f"- `{Path(s).name}`" if "\\" in s or "/" in s else f"- {s}")

    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, TXT, MD)",
        type=["pdf", "txt", "md", "csv", "json"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for file in uploaded_files:
            file_path = config.uploads_dir / file.name
            if not file_path.exists():
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                with st.spinner(f"Ingesting {file.name}..."):
                    chunks = ingest_file(str(file_path))
                    st.success(f"Indexed {file.name} ({chunks} chunks)")
                st.rerun()

    st.divider()
    st.subheader("Paste Job Description / Text")
    raw_title = st.text_input("Title (e.g. Accenture Job Description):", value="Job Description")
    raw_text = st.text_area("Paste text directly here:", height=120)
    if st.button("Index Text") and raw_text.strip():
        with st.spinner("Indexing text snippet..."):
            doc = Document(page_content=raw_text.strip(), metadata={"source": raw_title, "type": "manual_text"})
            chunks = ingest_documents([doc])
            st.success(f"Indexed '{raw_title}' ({chunks} chunks)")
            st.rerun()

    st.divider()
    st.subheader("Index Webpage")
    url_input = st.text_input("Enter URL to scrape and index:")
    if st.button("Index URL") and url_input.strip():
        with st.spinner("Scraping and vectorizing..."):
            try:
                chunks = ingest_url(url_input.strip())
                st.success(f"Successfully indexed URL ({chunks} chunks)")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to index: {e}")

    st.divider()
    if st.button("Start Fresh (Clear All Data)", type="primary", use_container_width=True):
        from vector_store import reset_vector_store
        reset_vector_store()
        for f in config.uploads_dir.glob("*.*"):
            try:
                f.unlink()
            except Exception:
                pass
        st.session_state.messages = []
        st.success("Session reset! Knowledge base is completely fresh (0 documents).")
        st.rerun()

    if st.button("Clear Chat Only", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("AgentIQ: Multi-Agent Research Assistant")
st.write(
    "Orchestrates specialized agents (Researcher, WebSearcher, and Analyst) using LangGraph and ChromaDB."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a research question, request resume analysis, or query documents...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    active_sources = get_indexed_sources()
    sources_summary = ", ".join([Path(s).name for s in active_sources[:6]])
    enriched_query = f"{user_query}\n\n[Context: Available indexed files/sources in Knowledge Base: {sources_summary}]"

    with st.chat_message("assistant"):
        final_answer = ""
        agent_steps = []

        with st.status("Agents at work...", expanded=True) as status:
            for step in stream_workflow(enriched_query):
                if "next" in step:
                    action_msg = f"**Supervisor Router** -> Delegated to **{step['next']}**"
                    agent_steps.append(action_msg)
                    st.write(action_msg)
                elif "node" in step:
                    node = step["node"]
                    content = step.get("content", "")
                    sender = step.get("sender", node)
                    if node == "synthesizer":
                        final_answer = content
                    else:
                        step_summary = f"**{sender}** finished research step"
                        agent_steps.append(step_summary)
                        st.write(step_summary)
                        with st.expander(f"View details from {sender}"):
                            st.markdown(content)

            status.update(label="Research complete!", state="complete", expanded=False)

        if final_answer:
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
