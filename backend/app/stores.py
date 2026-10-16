import chromadb
from langchain_anthropic import ChatAnthropic
from langchain_neo4j import Neo4jGraph

from . import config


def get_llm() -> ChatAnthropic:
    return ChatAnthropic(model=config.LLM_MODEL, temperature=0)


def get_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
    )


def get_collection():
    # Chroma's default embedding function (local MiniLM) -> no extra API key.
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return client.get_or_create_collection("chunks")
