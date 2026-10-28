from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.models import TranscriptSegment


def build_collection(meeting_id: str, transcript: str, segments: list[TranscriptSegment], persist_dir: Path):
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=f"meeting_{meeting_id}",
        embedding_function=SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2"),
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count():
        return collection
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_text(transcript)
    # Timestamp mapping is approximate for chunks, precise where Whisper segments are available.
    metadatas = []
    for index, chunk in enumerate(chunks):
        start = next((s.label for s in segments if s.text and s.text[:30] in chunk), "00:00")
        metadatas.append({"chunk": index + 1, "timestamp": start})
    collection.add(ids=[f"chunk-{i}" for i in range(len(chunks))], documents=chunks, metadatas=metadatas)
    return collection


def retrieve(collection, question: str, n_results: int = 4) -> tuple[str, list[dict]]:
    result = collection.query(query_texts=[question], n_results=min(n_results, max(collection.count(), 1)))
    docs = result.get("documents", [[]])[0]
    metadata = result.get("metadatas", [[]])[0]
    context = "\n\n".join(f"[{m.get('timestamp', '00:00')}] {doc}" for doc, m in zip(docs, metadata))
    return context, metadata
