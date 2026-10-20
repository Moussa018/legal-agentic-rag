"""Routeur de mode : Jev (TypeSafe, modèle System One) choisit vector / hybrid / agentic.

Une seule requête HTTP envoie trois questions typées, évaluées en parallèle par Jev.
Le code combine ensuite les réponses (seuils ci-dessous). Toute erreur (clé absente,
réseau, délai, réponse inattendue) retombe sur "agentic", le mode le plus robuste.
"""
import os

import requests

TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
TYPESAFE_MODEL = os.getenv("TYPESAFE_MODEL", "jev-latest")
TYPESAFE_URL = os.getenv("TYPESAFE_URL", "https://api.typesafe.ai/v1/systemone")
ROUTER_TIMEOUT = float(os.getenv("ROUTER_TIMEOUT", "3"))

FALLBACK_MODE = "agentic"

# Seuils de départ : à calibrer sur un vrai jeu de questions (eval/questions.json).
GRAPH_YES = 0.65       # P(le graphe est nécessaire) >= : oui
GRAPH_NO = 0.35        # P(...) <= : non ; entre les deux : incertain -> agentic
HOPS_AGENTIC = 1.5     # score attendu de sauts >= : plusieurs sauts -> agentic
MIN_CONFIDENCE = 0.5   # confiance minimale du score de sauts, sinon -> agentic
EXACT_TERM_YES = 0.6   # P(terme exact à retrouver) >= : on veut BM25 (hybrid)

# Questions atomiques : chacune ne juge qu'un seul point, le code les combine.
QUESTIONS = {
    "needs_graph": {
        "type": "noul",
        "instructions": (
            "Pour répondre à cette question juridique, faut-il suivre des relations "
            "entre entités (sociétés, personnes, contrats, clauses, dates, lois) ?"
        ),
        "criteria": {
            "true": "La réponse exige de relier plusieurs entités (qui est partie à quel contrat, "
                    "quelle clause référence quelle loi, quelle date d'expiration).",
            "false": "La réponse se trouve dans un passage de texte, sans relier d'entités.",
        },
    },
    "hops": {
        "type": "score",
        "instructions": "Combien de relations faut-il enchaîner pour répondre à la question ?",
        "criteria": [
            "Aucune relation, ou un seul fait",
            "Une relation entre deux entités",
            "Plusieurs relations enchaînées, comparaison ou agrégation",
        ],
    },
    "exact_term": {
        "type": "noul",
        "instructions": (
            "La question cite-t-elle un terme exact à retrouver tel quel "
            "(numéro de clause ou d'article, référence de loi, nom propre, montant) ?"
        ),
    },
}


def _evaluate(question: str) -> dict:
    resp = requests.post(
        TYPESAFE_URL,
        headers={"Authorization": f"Bearer {TYPESAFE_API_KEY}"},
        json={"model": TYPESAFE_MODEL, "state": question, "questions": QUESTIONS},
        timeout=ROUTER_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["answers"]


def decide(answers: dict) -> tuple[str, str]:
    """Combine les réponses de Jev en un mode + une raison lisible."""
    graph = answers["needs_graph"]["noul"]
    hops = answers["hops"]
    exact = answers["exact_term"]["noul"]

    if GRAPH_NO < graph < GRAPH_YES:
        return "agentic", f"besoin du graphe incertain (p={graph:.2f})"

    if graph >= GRAPH_YES:
        if hops["score"] >= HOPS_AGENTIC:
            return "agentic", f"graphe requis, plusieurs sauts (score={hops['score']:.2f})"
        if hops["confidence"] < MIN_CONFIDENCE:
            return "agentic", f"nombre de sauts incertain (confiance={hops['confidence']:.2f})"
        return "hybrid", f"graphe requis, peu de sauts (score={hops['score']:.2f})"

    if exact >= EXACT_TERM_YES:
        return "hybrid", f"pas de graphe mais terme exact à retrouver (p={exact:.2f})"
    return "vector", f"question textuelle simple (graphe p={graph:.2f})"


def route(question: str) -> dict:
    """Retourne {"mode", "reason", "answers"}. Ne lève jamais d'exception."""
    if not TYPESAFE_API_KEY:
        return {"mode": FALLBACK_MODE, "reason": "TYPESAFE_API_KEY absente", "answers": None}
    try:
        answers = _evaluate(question)
        mode, reason = decide(answers)
        return {"mode": mode, "reason": reason, "answers": answers}
    except Exception as e:  # réseau, délai, 4xx/5xx, JSON ou clés inattendus
        return {"mode": FALLBACK_MODE, "reason": f"routeur indisponible ({type(e).__name__})", "answers": None}
