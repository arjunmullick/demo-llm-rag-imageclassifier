import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any

from openai import OpenAI

from .vectorstore import SimpleVectorStore


DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


@dataclass
class RAGConfig:
    data_dir: Path
    index_path: Path
    embed_model: str = DEFAULT_EMBED_MODEL
    chat_model: str = DEFAULT_CHAT_MODEL


class RAGPipeline:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.client = OpenAI()
        self.store = SimpleVectorStore(cfg.index_path)
        self.store.load()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # The OpenAI SDK supports batching; keep batches small to avoid limits
        embeddings: List[List[float]] = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self.client.embeddings.create(model=self.cfg.embed_model, input=batch)
            for d in resp.data:
                embeddings.append(d.embedding)
        return embeddings

    def ingest_records(self, records: Iterable[Dict[str, Any]], source_name: str) -> int:
        # Flatten records into chunks
        chunks: List[str] = []
        ids: List[str] = []
        for rec in records:
            text = rec.get("text") or rec.get("description") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            for ch in chunk_text(text):
                chunks.append(ch)
                ids.append(str(uuid.uuid4()))

        if not chunks:
            return 0

        embeds = self.embed_texts(chunks)
        for doc_id, ch, emb in zip(ids, chunks, embeds):
            self.store.add(doc_id, ch, source_name, emb)
        self.store.save()
        return len(chunks)

    def ensure_index_built(self) -> None:
        # If no items, try building from sample dataset
        if not self.store.items:
            sample_path = self.cfg.data_dir / "sample_imaging.jsonl"
            if sample_path.exists():
                records = []
                with sample_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            import json
                            records.append(json.loads(line))
                        except Exception:
                            pass
                if records:
                    self.ingest_records(records, source_name=str(sample_path.name))

    def retrieve_context(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        q_emb = self.embed_texts([query])[0]
        return self.store.search(q_emb, k=k)

    def build_prompt(self, query: str, contexts: List[Dict[str, Any]], history: List[Dict[str, str]] | None = None) -> List[Dict[str, str]]:
        context_block = "\n\n".join([f"[Source: {c['source']}]\n{c['text']}" for c in contexts])
        system = (
            "You are a helpful assistant answering questions about medical imaging "
            "and radiology. Use only the provided context. If unsure, say you don't know."
        )
        user = (
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n"
        )
        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        if history:
            # keep the last ~6 messages to control prompt size
            for m in history[-6:]:
                if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
                    messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user})
        return messages

    def prepare_messages(self, query: str, k: int = 4, history: List[Dict[str, str]] | None = None):
        self.ensure_index_built()
        ctxs = self.retrieve_context(query, k=k)
        messages = self.build_prompt(query, ctxs, history=history)
        return messages, ctxs

    def chat(self, query: str, k: int = 4, history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
        messages, ctxs = self.prepare_messages(query, k=k, history=history)
        resp = self.client.chat.completions.create(
            model=self.cfg.chat_model,
            messages=messages,
            temperature=0.2,
        )
        answer = resp.choices[0].message.content
        return {
            "answer": answer,
            "contexts": [{"text": c["text"], "source": c["source"]} for c in ctxs],
            "model": self.cfg.chat_model,
        }
