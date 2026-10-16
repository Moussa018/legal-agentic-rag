import os

from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme123")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
PDF_DIR = os.getenv("PDF_DIR", "./data/pdfs")

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
