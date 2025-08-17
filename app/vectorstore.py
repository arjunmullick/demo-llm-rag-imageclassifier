import json
import math
from pathlib import Path
from typing import List, Dict, Any


def cosine_similarity(a: List[float], b: List[float]) -> float:
    # Safe cosine for lists
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-8
    nb = math.sqrt(sum(y * y for y in b)) or 1e-8
    return dot / (na * nb)


class SimpleVectorStore:
    def __init__(self, path: Path):
        self.path = path
        self.items: List[Dict[str, Any]] = []

    def load(self) -> None:
        if self.path.exists():
            self.items = json.loads(self.path.read_text())
        else:
            self.items = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.items, ensure_ascii=False, indent=2))

    def add(self, doc_id: str, text: str, source: str, embedding: List[float]) -> None:
        self.items.append({
            "id": doc_id,
            "text": text,
            "source": source,
            "embedding": embedding,
        })

    def search(self, query_embedding: List[float], k: int = 4) -> List[Dict[str, Any]]:
        scored = []
        for item in self.items:
            score = cosine_similarity(query_embedding, item["embedding"]) if item.get("embedding") else 0.0
            scored.append((score, item))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [item for _, item in scored[:k]]

