# Chat with your PDFs (RAG)

A retrieval-augmented generation app. Upload PDFs, ask questions, and get answers
grounded in the documents, with the source file and page shown for every answer.

## How it works

1. **Ingest:** each PDF page is split into overlapping chunks (LangChain `RecursiveCharacterTextSplitter`, 800 characters, 100 overlap).
2. **Embed and store:** chunks are embedded with `all-MiniLM-L6-v2` (sentence-transformers) and stored in ChromaDB.
3. **Retrieve:** a question is embedded and the top-k most similar chunks are fetched by cosine similarity.
4. **Generate:** the chunks and question go to an LLM (Anthropic or OpenAI) that answers only from that context and cites sources. With `LLM_PROVIDER=none`, the app shows the retrieved passages instead.

## Stack

Python, FastAPI, ChromaDB, sentence-transformers, LangChain (text splitter), HTML + Tailwind CSS, Docker.

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then set LLM_PROVIDER and your API key
uvicorn app.main:app --reload
```

Open http://localhost:8000, upload a PDF and ask a question.

## Run with Docker

```bash
docker build -t rag-pdf-chat .
docker run -p 8000:8000 --env-file .env rag-pdf-chat
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/upload` | Upload and index a PDF |
| POST | `/ask` | Ask a question; returns answer, sources, latency |
| GET | `/documents` | List indexed documents |
| GET | `/health` | Health check |

## Evaluation

`evaluate.py` measures retrieval hit rate (is the correct page in the top-4 results?)
and median retrieval latency on a hand-labelled question set.

```bash
python ingest.py                                  # index PDFs in ./data
cp eval_questions.example.json eval_questions.json  # add your questions
python evaluate.py
```

Results on my test set: _fill in after running (documents, number of questions, hit rate, median latency)._

## Limitations

- Scanned PDFs without a text layer are not supported (no OCR).
- Fixed-size chunking; no reranking yet.
- Single collection shared by all uploads.
