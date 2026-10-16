"""PDF -> chunks -> (Chroma vectors) + (LLM-extracted entities/relations -> Neo4j).

Run: python -m app.ingest
"""
from pathlib import Path

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from . import config
from .stores import get_collection, get_graph, get_llm


def load_chunks(pdf_path: Path) -> list[Document]:
    reader = PdfReader(str(pdf_path))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for part in splitter.split_text(text):
            docs.append(
                Document(page_content=part, metadata={"source": pdf_path.name, "page": page_no})
            )
    return docs


def ingest_file(pdf_path: Path, transformer: LLMGraphTransformer, graph, collection) -> int:
    chunks = load_chunks(pdf_path)
    if not chunks:
        return 0

    ids = [f"{pdf_path.name}:{i}" for i in range(len(chunks))]
    collection.upsert(
        ids=ids,
        documents=[c.page_content for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )

    graph_docs = transformer.convert_to_graph_documents(chunks)
    # include_source links each entity to its Document node -> lets answers cite sources.
    graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)
    return len(chunks)


def main() -> None:
    graph = get_graph()
    collection = get_collection()
    transformer = LLMGraphTransformer(
        llm=get_llm(),
        allowed_nodes=config.ALLOWED_NODES,
        allowed_relationships=config.ALLOWED_RELATIONSHIPS,
        node_properties=["nom"],
        relationship_properties=False,
    )

    pdfs = sorted(Path(config.PDF_DIR).glob("*.pdf"))
    if not pdfs:
        print(f"Aucun PDF dans {config.PDF_DIR}")
        return
    for pdf in pdfs:
        n = ingest_file(pdf, transformer, graph, collection)
        print(f"{pdf.name}: {n} chunks")
    graph.refresh_schema()


if __name__ == "__main__":
    main()
