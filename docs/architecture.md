# Imaging RAG + Image Classifier — Architecture

This document outlines the high-level components and request flows for the demo.

Pre-rendered images (SVG):

- Overview: architecture_overview.svg
- RAG Ingest: rag_ingest.svg
- RAG Query (Streaming): rag_query_stream.svg
- Image Classifier: image_classifier.svg

## Components

```mermaid
flowchart LR
  subgraph Browser[Browser UI]
    UI[(Static HTML/JS)]
  end

  subgraph Server[FastAPI Server]
    API1[/POST /ingest/]
    API2[/POST /chat/ & GET /chat/stream/]
    API3[/POST /classify_image/]
    RAG[RAG Pipeline: chunk, embed, retrieve, prompt]
    VS[(JSON Vector Index: data/index.json)]
  end

  subgraph OpenAI[OpenAI APIs]
    E[Embeddings]
    C[Chat (Text + Vision)]
  end

  UI -->|Build Index| API1
  UI -->|Ask (Stream)| API2
  UI -->|Upload Image| API3

  API1 --> RAG
  API2 --> RAG
  RAG <--> VS
  RAG -->|embed/query| E
  RAG -->|answer| C
  API3 -->|vision classify| C
```

## RAG — Ingestion Flow

```mermaid
sequenceDiagram
  participant UI as Browser UI
  participant API as FastAPI /ingest
  participant RAG as RAG Pipeline
  participant E as OpenAI Embeddings
  participant VS as Vector Index (JSON)

  UI->>API: POST /ingest
  API->>RAG: load & chunk dataset
  RAG->>E: create embeddings (batched)
  E-->>RAG: vectors
  RAG->>VS: save items (text, source, embedding)
  API-->>UI: {added_chunks, index_path}
```

## RAG — Query Flow (Streaming)

```mermaid
sequenceDiagram
  participant UI as Browser UI
  participant API as FastAPI /chat/stream
  participant RAG as RAG Pipeline
  participant E as OpenAI Embeddings
  participant C as OpenAI Chat

  UI->>API: GET /chat/stream?message=...
  API->>RAG: prepare_messages(message, k, history)
  RAG->>E: embed(query)
  RAG->>RAG: cosine retrieve top-k
  RAG->>C: chat.completions (with context)
  C-->>API: streamed tokens (SSE)
  API-->>UI: data: {type: token, content: ...}
  API-->>UI: data: {type: end, contexts, model}
```

## Image Classifier — Basic and Few‑Shot

```mermaid
sequenceDiagram
  participant UI as Browser UI
  participant API as FastAPI /classify_image
  participant C as OpenAI Chat (Vision)

  UI->>API: POST multipart (file[, few_shot, example_cat, example_dog])
  API->>API: build messages
  alt few_shot enabled
    API->>API: add cat example -> assistant: "cat"
    API->>API: add dog example -> assistant: "dog"
  end
  API->>C: chat.completions (text + image)
  C-->>API: label text
  API->>API: normalize -> {cat|dog|unknown}
  API-->>UI: {label, raw}
```

## Notes

- Vector store is a simple JSON file, suitable for small demos. Swap with FAISS/Chroma/pgvector for scale.
- Sessions are in-memory; use Redis or DB for multi-instance deployments.
- Models default to `gpt-4o-mini` (chat/vision) and `text-embedding-3-small` (embeddings).
