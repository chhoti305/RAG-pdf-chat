"""FastAPI app: upload PDFs, ask questions, get answers with sources."""
import os
import shutil
import time
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import llm, rag

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Chat with your PDFs (RAG)")
app.mount("/static", StaticFiles(directory="static"), name="static")


class AskRequest(BaseModel):
    question: str
    top_k: int = int(os.getenv("TOP_K", "4"))


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def documents():
    return rag.list_documents()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    safe_name = os.path.basename(file.filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    with open(path, "wb") as out:
        shutil.copyfileobj(file.file, out)
    try:
        chunks = rag.ingest_pdf(path, safe_name)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {exc}")
    if chunks == 0:
        raise HTTPException(
            status_code=422, detail="No text found. Scanned PDFs are not supported."
        )
    return {"file": safe_name, "chunks_stored": chunks}


@app.post("/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    t0 = time.perf_counter()
    chunks = rag.retrieve(question, k=req.top_k)
    t1 = time.perf_counter()
    try:
        answer = llm.generate_answer(question, chunks)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}")
    t2 = time.perf_counter()

    return {
        "answer": answer,
        "sources": chunks,
        "latency_ms": {
            "retrieval": round((t1 - t0) * 1000),
            "generation": round((t2 - t1) * 1000),
        },
    }
