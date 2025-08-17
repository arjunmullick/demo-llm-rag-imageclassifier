import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from .rag import RAGConfig, RAGPipeline


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_PATH = DATA_DIR / "index.json"

app = FastAPI(title="Imaging RAG Demo")


def make_pipeline() -> RAGPipeline:
    cfg = RAGConfig(data_dir=DATA_DIR, index_path=INDEX_PATH)
    return RAGPipeline(cfg)


class IngestRequest(BaseModel):
    # Optional JSONL path relative to data/
    jsonl_name: Optional[str] = None


@app.get("/")
def home() -> HTMLResponse:
    html = (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/ingest")
def ingest(req: IngestRequest):
    import json
    p = make_pipeline()
    p.store.load()
    path = None
    if req.jsonl_name:
        path = DATA_DIR / req.jsonl_name
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
    else:
        path = DATA_DIR / "sample_imaging.jsonl"

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    added = p.ingest_records(records, source_name=str(path.name))
    return {"added_chunks": added, "index_path": str(INDEX_PATH)}


class ChatRequest(BaseModel):
    message: str
    k: int = 4
    session_id: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str


# naive in-memory session store for demo purposes
CHAT_SESSIONS: dict[str, list[dict[str, str]]] = {}


@app.post("/session", response_model=SessionResponse)
def create_session() -> SessionResponse:
    import uuid
    sid = str(uuid.uuid4())
    CHAT_SESSIONS[sid] = []
    return SessionResponse(session_id=sid)


@app.post("/chat")
def chat(req: ChatRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")
    p = make_pipeline()
    try:
        history = CHAT_SESSIONS.get(req.session_id) if req.session_id else None
        result = p.chat(req.message, k=req.k, history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # update session history
    if req.session_id:
        CHAT_SESSIONS.setdefault(req.session_id, [])
        CHAT_SESSIONS[req.session_id].append({"role": "user", "content": req.message})
        CHAT_SESSIONS[req.session_id].append({"role": "assistant", "content": result["answer"]})
    return result


@app.get("/chat/stream")
def chat_stream(
    message: str = Query(..., description="User message"),
    k: int = Query(4, ge=1, le=10),
    session_id: Optional[str] = Query(None),
):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")

    import json
    p = make_pipeline()

    # prepare messages and contexts (includes RAG retrieval)
    history = CHAT_SESSIONS.get(session_id) if session_id else None
    try:
        messages, contexts = p.prepare_messages(message, k=k, history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    client = p.client

    def event_gen():
        full_answer = []
        try:
            stream = client.chat.completions.create(
                model=p.cfg.chat_model,
                messages=messages,
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    full_answer.append(token)
                    yield f"data: {json.dumps({'type':'token','content':token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','error':str(e)})}\n\n"
            return

        answer = "".join(full_answer)
        # update session history
        if session_id:
            CHAT_SESSIONS.setdefault(session_id, [])
            CHAT_SESSIONS[session_id].append({"role": "user", "content": message})
            CHAT_SESSIONS[session_id].append({"role": "assistant", "content": answer})

        payload = {
            "type": "end",
            "answer": answer,
            "contexts": [{"text": c["text"], "source": c["source"]} for c in contexts],
            "model": p.cfg.chat_model,
        }
        yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/classify_image")
def classify_image(
    file: UploadFile = File(...),
    few_shot: bool = Form(False),
    example_cat: UploadFile | None = File(None),
    example_dog: UploadFile | None = File(None),
):
    import base64
    import mimetypes

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")

    # Read file bytes
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    # Determine MIME type
    mime = file.content_type or mimetypes.guess_type(file.filename or "")[0] or "image/jpeg"
    b64 = base64.b64encode(data).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    client = make_pipeline().client  # reuse configured OpenAI client

    system = "Classify images as 'cat' or 'dog'. If unsure, answer 'unknown'. Respond with one word."

    # build optional few-shot examples
    examples: list[dict] = []
    def file_to_data_url(up: UploadFile | None) -> str | None:
        if not up:
            return None
        b = up.file.read()
        if not b:
            return None
        m = up.content_type or mimetypes.guess_type(up.filename or "")[0] or "image/jpeg"
        b64 = base64.b64encode(b).decode("ascii")
        return f"data:{m};base64,{b64}"

    if few_shot:
        cat_url = file_to_data_url(example_cat)
        dog_url = file_to_data_url(example_dog)
        if cat_url:
            examples.append({"role":"user","content":[{"type":"text","text":"Example: classify this image."},{"type":"image_url","image_url":{"url":cat_url}}]})
            examples.append({"role":"assistant","content":"cat"})
        if dog_url:
            examples.append({"role":"user","content":[{"type":"text","text":"Example: classify this image."},{"type":"image_url","image_url":{"url":dog_url}}]})
            examples.append({"role":"assistant","content":"dog"})

    messages = [{"role":"system","content":system}] + examples + [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Classify this image. Answer: cat, dog, or unknown."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    try:
        resp = client.chat.completions.create(
            model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0,
        )
        label = (resp.choices[0].message.content or "").strip().lower()
        if "cat" in label and "dog" in label:
            final = "unknown"
        elif "cat" in label:
            final = "cat"
        elif "dog" in label:
            final = "dog"
        else:
            final = "unknown"
        return {"label": final, "raw": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
