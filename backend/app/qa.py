"""Vector-only RAG vs hybrid GraphRAG (vector + Cypher traversal)."""
import re

from langchain_core.prompts import ChatPromptTemplate

from . import bm25
from .stores import get_graph, get_llm

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Tu es un assistant juridique. Réponds uniquement à partir du contexte.
Si le contexte est insuffisant, dis-le. Cite les sources (fichier, page) quand elles existent.

Contexte documentaire :
{vector_context}

Faits issus du graphe :
{graph_context}

Question : {question}
Réponse :"""
)

CYPHER_PROMPT = ChatPromptTemplate.from_template(
    """Génère UNE requête Cypher en lecture seule (MATCH/RETURN, jamais de CREATE/MERGE/DELETE/SET)
pour répondre à la question, d'après ce schéma de graphe :
{schema}

Les noms d'entités sont dans la propriété `id`. Utilise toLower(...) CONTAINS pour les correspondances.
Les dates sont des nœuds Date dont `id` est une chaîne. Limite à 50 lignes.
Réponds avec la requête seule, sans markdown.

Question : {question}
Cypher :"""
)

WRITE_KEYWORDS = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL|LOAD)\b", re.I)


def vector_search(question: str, k: int = 5) -> list[dict]:
    """Pure dense retrieval (baseline)."""
    return bm25.vector_hits(question, k)


def retrieve(question: str, k: int = 5) -> list[dict]:
    """Dense + BM25 fused with RRF. Used by hybrid and agentic modes."""
    return bm25.fused_search(question, k)


def graph_search(question: str) -> tuple[str, list[dict]]:
    graph = get_graph()
    llm = get_llm()
    cypher = (CYPHER_PROMPT | llm).invoke({"schema": graph.schema, "question": question}).content
    cypher = cypher.strip().strip("`").removeprefix("cypher").strip()
    if WRITE_KEYWORDS.search(cypher):
        return cypher, []
    try:
        return cypher, graph.query(cypher)
    except Exception:
        # LLM-written Cypher can be invalid; fall back to vector-only context.
        return cypher, []


def _format_chunks(chunks: list[dict]) -> str:
    return "\n---\n".join(f"[{c['source']} p.{c['page']}] {c['text']}" for c in chunks)


def answer(question: str, mode: str = "hybrid") -> dict:
    chunks = vector_search(question) if mode == "vector" else retrieve(question)
    cypher, rows = ("", [])
    if mode == "hybrid":
        cypher, rows = graph_search(question)

    msg = (ANSWER_PROMPT | get_llm()).invoke(
        {
            "vector_context": _format_chunks(chunks),
            "graph_context": str(rows) if rows else "(aucun)",
            "question": question,
        }
    )
    return {
        "answer": msg.content,
        "mode": mode,
        "cypher": cypher,
        "graph_rows": rows,
        "sources": [{"source": c["source"], "page": c["page"]} for c in chunks],
    }
