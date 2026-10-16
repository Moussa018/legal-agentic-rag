"""Agentic RAG: the LLM decides which tools to call (and how many times) before answering.

Unlike qa.answer (fixed pipeline: one vector search + one Cypher), the agent can
inspect the schema, run several Cypher queries, retry after errors, and refine searches.
"""
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from .bm25 import bm25_search
from .qa import WRITE_KEYWORDS, retrieve, vector_search
from .stores import get_graph, get_llm

MAX_STEPS = 8

SYSTEM = """Tu es un assistant juridique agentique. Tu réponds à partir de documents (PDF) et d'un graphe Neo4j.
Méthode :
1. Pour une question sur des relations entre entités, commence par get_graph_schema, puis run_cypher.
2. Pour retrouver le texte et citer les sources, choisis l'outil de recherche adapté : keyword_search pour un terme exact
   (n° de clause, nom, loi), semantic_search pour un concept, search_documents par défaut. Si le premier ne trouve rien,
   essaie un autre outil ou reformule. Utilise les entités trouvées dans le graphe comme mots-clés pour keyword_search.
3. Si une requête Cypher échoue ou renvoie 0 ligne, corrige-la ou reformule la recherche. Ne répète pas la même requête.
4. Décompose les questions multi-sauts en étapes.
Réponds uniquement avec ce que les outils ont renvoyé ; si c'est insuffisant, dis-le. Cite fichier et page."""


def _fmt(chunks: list[dict]) -> str:
    return "\n---\n".join(f"[{c['source']} p.{c['page']}] {c['text']}" for c in chunks) or "Aucun résultat."


@tool
def search_documents(query: str) -> str:
    """Recherche hybride (sémantique + BM25 fusionnés). Choix par défaut pour une question formulée en langage naturel."""
    return _fmt(retrieve(query, k=5))


@tool
def keyword_search(terms: str) -> str:
    """Recherche BM25 par mots-clés exacts : n° de clause ou d'article, nom propre, référence de loi, montant.
    Donne quelques termes précis, pas une phrase. À préférer quand search_documents rate un terme exact."""
    return _fmt(bm25_search(terms, k=5))


@tool
def semantic_search(query: str) -> str:
    """Recherche vectorielle pure : pour une notion ou un concept sans vocabulaire précis (ex. « obligation de confidentialité »)."""
    return _fmt(vector_search(query, k=5))


@tool
def get_graph_schema() -> str:
    """Renvoie le schéma du graphe Neo4j (labels, propriétés, relations)."""
    return get_graph().schema


@tool
def run_cypher(query: str) -> str:
    """Exécute une requête Cypher en LECTURE SEULE sur le graphe. Les noms d'entités sont dans la propriété `id`."""
    if WRITE_KEYWORDS.search(query):
        return "Refusé : seules les requêtes en lecture (MATCH/RETURN) sont autorisées."
    try:
        rows = get_graph().query(query)
    except Exception as e:  # let the agent see the error and fix its query
        return f"Erreur Cypher : {e}"
    return str(rows[:50]) if rows else "0 ligne."


TOOLS = {
    t.name: t
    for t in (search_documents, keyword_search, semantic_search, get_graph_schema, run_cypher)
}


def run_agent(question: str) -> dict:
    llm = get_llm().bind_tools(list(TOOLS.values()))
    messages = [SystemMessage(SYSTEM), HumanMessage(question)]
    trace = []

    for _ in range(MAX_STEPS):
        ai = llm.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return {"answer": ai.content, "mode": "agentic", "steps": trace}
        for call in ai.tool_calls:
            result = TOOLS[call["name"]].invoke(call["args"])
            trace.append({"tool": call["name"], "args": call["args"], "result": str(result)[:500]})
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    # Step budget exhausted: force a final answer from what was gathered.
    final = get_llm().invoke(messages + [HumanMessage("Donne ta meilleure réponse avec les informations obtenues.")])
    return {"answer": final.content, "mode": "agentic", "steps": trace, "truncated": True}
