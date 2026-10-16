import os

from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme123")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
PDF_DIR = os.getenv("PDF_DIR", "./data/pdfs")

# Two-stage retrieval: stage 1 fetches RERANK_CANDIDATES chunks, a cross-encoder keeps the best k.
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in ("1", "true", "yes")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")  # multilingual
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))

# Closed schema keeps the graph consistent across documents.
ALLOWED_NODES = ["Personne", "Societe", "Contrat", "Clause", "Date", "Loi"]
ALLOWED_RELATIONSHIPS = [
    ("Societe", "PARTIE_A", "Contrat"),
    ("Personne", "PARTIE_A", "Contrat"),
    ("Personne", "REPRESENTE", "Societe"),
    ("Contrat", "CONTIENT", "Clause"),
    ("Contrat", "EXPIRE_LE", "Date"),
    ("Contrat", "SIGNE_LE", "Date"),
    ("Clause", "REFERENCE", "Loi"),
    ("Societe", "FILIALE_DE", "Societe"),
]
