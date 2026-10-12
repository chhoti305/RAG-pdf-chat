"""Ingestion and retrieval: PDF -> chunks -> embeddings -> ChromaDB."""
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

_collection = None


def get_collection():
    """Create the vector store on first use (loading the embedding model is slow)."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        _collection = client.get_or_create_collection(
            name="documents",
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def ingest_pdf(path: str, source_name: str) -> int:
    """Read a PDF, split each page into chunks, store them. Returns chunk count."""
    collection = get_collection()
    # Re-uploading the same file replaces its old chunks.
    collection.delete(where={"source": source_name})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    reader = PdfReader(path)

    ids, texts, metas = [], [], []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        for i, chunk in enumerate(splitter.split_text(text)):
            ids.append(f"{source_name}-p{page_no}-c{i}")
            texts.append(chunk)
            metas.append({"source": source_name, "page": page_no})

    if ids:
        collection.add(ids=ids, documents=texts, metadatas=metas)
    return len(ids)


def retrieve(question: str, k: int = 4):
    """Return the top-k most similar chunks for a question."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    result = collection.query(query_texts=[question], n_results=k)
    chunks = []
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        chunks.append(
            {
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "score": round(1 - dist, 4),  # cosine similarity
            }
        )
    return chunks


def list_documents():
    collection = get_collection()
    data = collection.get(include=["metadatas"])
    counts = {}
    for meta in data["metadatas"]:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return [{"source": s, "chunks": c} for s, c in sorted(counts.items())]
