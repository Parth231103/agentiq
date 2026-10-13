import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

load_dotenv()


class LLMConfig(BaseSettings):
    provider: str = "openai"
    openai_api_key: str = Field(default="freellmapi-1e5e64ac371b05264151329356675dc3c535c47980b22a8f")
    openai_base_url: str = Field(default="http://127.0.0.1:31415/v1")
    chat_model: str = Field(default="auto")
    embedding_model: str = Field(default="text-embedding-3-small")
    temperature: float = 0.1
    max_tokens: int = 4096
    streaming: bool = True


class VectorStoreConfig(BaseSettings):
    collection_name: str = "agentiq_docs"
    persist_directory: str = "./data/chroma_db"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5


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


class AppConfig(BaseSettings):
    llm: LLMConfig = LLMConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    memory: MemoryConfig = MemoryConfig()
    agent: AgentConfig = AgentConfig()

    app_name: str = "AgentIQ"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    data_dir: Path = Path("./data")
    uploads_dir: Path = Path("./data/uploads")

    def setup_directories(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        Path(self.vector_store.persist_directory).mkdir(parents=True, exist_ok=True)


config = AppConfig()
config.setup_directories()


if __name__ == "__main__":
    print(f" {config.app_name} v{config.app_version}")
    print(f" Provider: {config.llm.provider}")
    print(f" Base URL: {config.llm.openai_base_url}")
    print(f" Chat Model: {config.llm.chat_model}")
    print(f" Embedding Model: {config.llm.embedding_model}")
    print(f"  Vector DB: {config.vector_store.persist_directory}")
    print(f" Chunk Size: {config.vector_store.chunk_size}")
    print(f" Top-K: {config.vector_store.top_k}")
    print(f" Memory Turns: {config.memory.max_conversation_turns}")

    if config.llm.openai_api_key:
        print(f" API Key: ***...{config.llm.openai_api_key[-4:]}")
    else:
        print("  No API key found! Add OPENAI_API_KEY to .env")
    