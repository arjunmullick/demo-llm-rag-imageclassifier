**Imaging RAG Demo**

- **Goal:** Quick local demo that retrieves from an imaging dataset and answers with an LLM using RAG.
- **Stack:** FastAPI + OpenAI Chat/Embeddings + simple JSON vector store + static HTML UI.
- **Context** This demo is a hands‑on, local showcase of retrieval‑augmented generation (RAG) and vision classification. It ships with a small
imaging dataset, builds a lightweight vector index using OpenAI embeddings, and serves a FastAPI backend with a minimal web UI.
Users can ingest the dataset, ask questions that retrieve relevant report snippets, and receive grounded, streamed answers from a
chat model with visible context. A companion image classifier lets you upload a photo to label “cat” or “dog,” optionally guided
by few‑shot example images—no training required. Everything runs locally, is simple to customize, and illustrates modern cloud AI
patterns end‑to‑end practically.

**Prereqs**

- Python 3.10+
- OpenAI API key in env: `OPENAI_API_KEY`

**Setup**

1) Create and activate a virtualenv (recommended)

   - `python -m venv .venv && source .venv/bin/activate`

2) Install dependencies

   - `pip install -r requirements.txt`

3) Configure env

   - Copy `.env.example` to `.env` and fill `OPENAI_API_KEY`, or export in your shell:
     - `export OPENAI_API_KEY=sk-...`

**Run**

- Start the API + UI
  - `uvicorn app.main:app --reload`
  - Open http://127.0.0.1:8000 in a browser.

- Click “Build/Refresh Index” to embed the bundled sample dataset (`data/sample_imaging.jsonl`). On first chat, the server will also attempt to build the index automatically if empty.

**Streaming + Sessions**

- The UI uses Server-Sent Events to stream token-by-token responses from `/chat/stream`.
- A session is created on first load (`/session`), and the `session_id` is stored in `localStorage` so chat history is preserved while the server stays running.

**How it works**

- `data/sample_imaging.jsonl`: Small imaging report-like texts.
- `POST /ingest`: Chunks records and stores embeddings to `data/index.json`.
- `POST /chat`: Embeds the query, retrieves top-k chunks by cosine similarity, and asks the chat model to answer using only that context.
- `static/index.html`: Minimal UI to call the endpoints.
  - Cat vs Dog: Upload an image to the "Cat vs Dog Classifier" section to classify using the OpenAI vision model.
  - Few-shot examples: Optionally provide one cat and one dog example to guide the classifier.

**Customize the Dataset**

- Add a new JSONL under `data/` with one JSON object per line containing a `text` field (and any metadata).
- Build the index for your file:
  - `curl -X POST -H 'Content-Type: application/json' \
     -d '{"jsonl_name":"your_file.jsonl"}' http://127.0.0.1:8000/ingest`

**Notes**

- Models default to `gpt-4o-mini` (chat) and `text-embedding-3-small` (embeddings). Override with `CHAT_MODEL`/`EMBED_MODEL` env vars.
- This demo stores a simple JSON vector index (`data/index.json`) for local use only. It’s not optimized for large datasets.
- Network access is required at runtime to call the OpenAI API.
- If you see `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`, run `pip install -r requirements.txt --force-reinstall` to ensure `httpx==0.27.2` is installed (compatibility with the OpenAI SDK version).

**Design & Architecture**

- See `docs/architecture.md` for Mermaid diagrams of:
  - High-level component architecture
  - RAG ingestion and query (streaming) flows
  - Image classifier request flow with optional few‑shot examples
  
  Pre-rendered images:
  - docs/architecture_overview.svg
  - docs/rag_ingest.svg
  - docs/rag_query_stream.svg
  - docs/image_classifier.svg

**Next Steps (optional)**

- Swap the vector store to FAISS, Chroma, or pgvector for scale.
- Expand dataset to full radiology reports; add basic metadata filters.
- Add streaming responses and chat history.

**Demo Explained**

- **Big picture:** Ask questions; the app retrieves the most relevant imaging snippets and the model answers grounded in that context (RAG).
- **Dataset:** Short radiology-like texts under `data/sample_imaging.jsonl`.
- **Indexing steps:**
  - Ingest: Load and chunk texts.
  - Embed: Convert chunks to vectors with the embeddings model.
  - Index: Store vectors locally in `data/index.json`.
- **Answering steps:**
  - Embed your question and retrieve top‑k similar chunks.
  - Build a prompt that includes only those chunks as context.
  - Call the chat model to answer using that context (with guardrails to avoid guessing).
  - Stream tokens to the UI; show the exact context snippets used.

**Good Test Prompts**

- Ground-glass: "What are ground-glass opacities and what condition are they compatible with in the CT chest case?"
- Lobar pneumonia: "Which report describes lobar pneumonia? Summarize the key radiographic signs."
- Knee MRI: "Summarize the key findings of the knee MRI case."
- Acute cholecystitis: "What ultrasound findings suggest acute cholecystitis in the dataset?"
- Head CT: "Is there any midline shift on the non-contrast head CT? What else is noted?"
- MS features: "What MRI brain features suggest multiple sclerosis in the dataset?"
- Hand X-ray: "What fracture is present on the hand X-ray and how is it described?"
- Obstetric US: "Summarize the 20-week obstetric ultrasound findings."
- Diverticulitis: "Describe the CT findings of sigmoid diverticulitis in the dataset."
- Lumbar spine: "Summarize the L4–L5 MRI lumbar spine findings."
- Compare pneumonias: "Compare the imaging findings of viral/atypical pneumonia vs lobar pneumonia in this dataset."
- Modality sweep: "List each case by modality and body part, with a one-line summary."
- Targeted sign: "Which report mentions a positive sonographic Murphy sign? What does it indicate?"
- Retrieval+guardrails: "Which case mentions mediastinal lymphadenopathy? Cite the source shown below."
- Out-of-scope test: "What is the patient’s name?" (Should say it's not in the context.)

**Troubleshooting**

- **Use venv Python:**
  - `source .venv/bin/activate`
  - Run with `python -m uvicorn app.main:app --reload` to ensure the venv interpreter is used.
  - Check: `which python` → should be `.venv/bin/python`.

- **Install/repair dependencies:**
  - `pip install -r requirements.txt --force-reinstall`
  - Verify: `python -c "import httpx, openai; print('httpx', httpx.__version__, 'openai', openai.__version__)"` → expect `httpx 0.27.x`, `openai 1.37.1`.

- **Missing FastAPI / ModuleNotFoundError: fastapi**
  - Likely using system Python. Activate the venv and reinstall deps.

- **TypeError 'proxies' (httpx/OpenAI mismatch)**
  - Cause: Newer `httpx` removed the `proxies` argument.
  - Fix: `pip install -r requirements.txt --force-reinstall` (this pins `httpx==0.27.2`).

- **OPENAI_API_KEY not set**
  - Set in `.env` or export in shell. Example `.env` line: `OPENAI_API_KEY=sk-...`
  - Restart the server after setting.

- **429 insufficient_quota**
  - Add billing or switch to a key with available credit: https://platform.openai.com/account/billing/overview
  - Try again after limits refresh.

- **Model access errors (404/permission)**
  - Override with env vars if needed: `export CHAT_MODEL=gpt-4o-mini`, `export EMBED_MODEL=text-embedding-3-small` (or `text-embedding-ada-002`).
  - Restart the server.

- **500 on /chat/stream or /ingest**
  - Check server logs for the exact exception.
  - Try non-streaming calls for clearer JSON errors:
    - `curl -s -X POST http://127.0.0.1:8000/ingest -H 'Content-Type: application/json' -d '{}'`
    - `curl -s -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' -d '{"message":"Tell me about the dataset","k":4}'`

- **Image classification errors**
  - Requires an API key with vision-enabled model access and sufficient quota.
  - Override chat model if needed: `export CHAT_MODEL=gpt-4o-mini`
  - Try a small JPG/PNG; very large images may be slow.

**Cat vs Dog Classifier (Few‑Shot) Usage**

- Basic classification:
  - Scroll to "Cat vs Dog Classifier", choose an image, click "Classify".
  - The server sends the image + a short instruction to the vision model; returns `cat`, `dog`, or `unknown`.
- With few‑shot examples (better guidance, no training required):
  - Tick "Use few-shot examples".
  - Add a representative cat image to "Cat example" and a dog image to "Dog example" (either or both).
  - Click "Classify" — the server includes those example images and their labels in the prompt before your image.
  - This is zero‑shot-with-examples (few‑shot prompting), not model training.

- **Port already in use**
  - `python -m uvicorn app.main:app --reload --port 8001`

- **Python version quirks**
  - The code targets Python 3.10+. If you need 3.8/3.9 support, replace `list | None` type hints with `Optional[list]` and install `typing_extensions`.
