from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import agent, qa

app = FastAPI(title="legal-agentic-rag")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    mode: str = "agentic"  # "agentic" | "hybrid" | "vector"


@app.post("/ask")
def ask(req: AskRequest):
    if req.mode == "agentic":
        return agent.run_agent(req.question)
    return qa.answer(req.question, req.mode)


@app.get("/health")
def health():
    return {"status": "ok"}
