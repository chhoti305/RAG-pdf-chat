"""Measure retrieval quality and speed.

Usage:
  1. Ingest your PDFs (python ingest.py).
  2. Copy eval_questions.example.json to eval_questions.json and fill it with
     15-20 questions whose answer location (file + page) you know.
  3. Run: python evaluate.py
"""
import json
import statistics
import time
from app import rag

K = 4
with open("eval_questions.json", encoding="utf-8") as f:
    items = json.load(f)

hits, latencies = 0, []
for item in items:
    t0 = time.perf_counter()
    chunks = rag.retrieve(item["question"], k=K)
    latencies.append((time.perf_counter() - t0) * 1000)
    found = any(
        c["source"] == item["source"] and c["page"] == item["page"] for c in chunks
    )
    hits += found
    print(("HIT  " if found else "MISS ") + item["question"])

print(f"\nQuestions: {len(items)}")
print(f"Hit rate @ top-{K}: {hits / len(items):.0%}")
print(f"Median retrieval latency: {statistics.median(latencies):.0f} ms")
