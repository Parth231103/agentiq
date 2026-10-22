import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field
from langchain_openai import ChatOpenAI

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent


class LLMConfig(BaseSettings):
    provider: str = Field(default="auto")
    openai_api_key: str | None = Field(default=None)
    openai_base_url: str | None = Field(default=None)
    chat_model: str = Field(default="auto")
    temperature: float = 0.1
    max_tokens: int = 4096
    streaming: bool = True


class VectorStoreConfig(BaseSettings):
    collection_name: str = "agentiq_docs"
    persist_directory: str = str(BASE_DIR / "data" / "chroma_db")
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    rrf_k: int = 60  # Reciprocal Rank Fusion smoothing constant


class MemoryConfig(BaseSettings):
    max_conversation_turns: int = 20
    long_term_collection: str = "agentiq_memory"
    memory_relevance_threshold: float = 0.7


class AgentConfig(BaseSettings):
    supervisor_name: str = "Supervisor"
    researcher_name: str = "Researcher"
    web_searcher_name: str = "WebSearcher"
    analyst_name: str = "Analyst"
    max_agent_steps: int = 10
    recursion_limit: int = 25
    enable_human_in_the_loop: bool = True


class AppConfig(BaseSettings):
    llm: LLMConfig = LLMConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    memory: MemoryConfig = MemoryConfig()
    agent: AgentConfig = AgentConfig()

    app_name: str = "AgentIQ"
    app_version: str = "1.1.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    data_dir: Path = BASE_DIR / "data"
    uploads_dir: Path = BASE_DIR / "data" / "uploads"

    def setup_directories(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        Path(self.vector_store.persist_directory).mkdir(parents=True, exist_ok=True)


config = AppConfig()
config.setup_directories()


def get_llm():
    """Centralized LLM factory with dynamic provider routing:
    Supports OpenRouter, Groq, OpenAI, or local OpenAI-compatible backends.
    Prioritizes explicit environment keys with zero hardcoded credentials.
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY") or config.llm.openai_api_key
    openai_base_url = os.getenv("OPENAI_BASE_URL") or config.llm.openai_base_url

    headers = {}
    if openrouter_key:
        api_key = openrouter_key
        base_url = "https://openrouter.ai/api/v1"
        model = os.getenv("OPENROUTER_MODEL") or os.getenv("CHAT_MODEL") or "nvidia/nemotron-3.5-lightning:free"
        headers = {
            "HTTP-Referer": "https://github.com/parth231103/agentiq",
            "X-Title": "AgentIQ"
        }
    elif groq_key and not openai_key:
        api_key = groq_key
        base_url = "https://api.groq.com/openai/v1"
        model = os.getenv("GROQ_MODEL") or os.getenv("CHAT_MODEL") or "llama-3.3-70b-versatile"
    elif openai_key:
        api_key = openai_key
        base_url = openai_base_url or "https://api.openai.com/v1"
        model = os.getenv("CHAT_MODEL") or config.llm.chat_model
    else:
        api_key = "local"
        base_url = openai_base_url or "http://127.0.0.1:31415/v1"
        model = os.getenv("CHAT_MODEL") or "auto"

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=config.llm.temperature,
        streaming=config.llm.streaming,
        default_headers=headers if headers else None,
    )


if __name__ == "__main__":
    print(f"AgentIQ Config initialized: {config.app_name} v{config.app_version}")
