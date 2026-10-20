from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import agent, qa, router

app = FastAPI(title="legal-agentic-rag")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    # "auto" : Jev choisit le mode ; les trois autres forcent un mode (utile pour l'évaluation).
    mode: Literal["auto", "agentic", "hybrid", "vector"] = "auto"


@app.post("/ask")
def ask(req: AskRequest):
    mode, routing = req.mode, None
    if mode == "auto":
        routing = router.route(req.question)
        mode = routing["mode"]

    if mode == "agentic":
        result = agent.run_agent(req.question)
    else:
        result = qa.answer(req.question, mode)

    if routing:
        result["routing"] = routing  # mode choisi, raison, réponses brutes de Jev
    return result


@app.get("/health")
def health():
    return {"status": "ok"}
