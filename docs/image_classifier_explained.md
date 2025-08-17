# Cat vs Dog Image Classifier — Explained

## Overview
- Purpose: A simple, local demo that classifies an uploaded image as “cat”, “dog”, or “unknown”.
- Approach: No training pipeline. Uses a pre‑trained OpenAI multimodal (vision) model via prompting.
- Extras: Optional few‑shot guidance by attaching 1–2 labeled example images (cat and/or dog) to improve consistency.

## What It Does (End‑to‑End Lifecycle)
1. User selects an image in the web UI and clicks “Classify”.
2. The browser sends a multipart/form‑data request to `POST /classify_image` (FastAPI).
3. The server reads the image bytes, detects MIME type, and builds a data URL (base64) for the model.
4. The server constructs a prompt:
   - System rule: “Classify images as cat/dog; if unsure, respond ‘unknown’.”
   - Optional few‑shot examples: A cat image labeled “cat” and/or a dog image labeled “dog”.
   - The image to classify.
5. The server calls the OpenAI Chat Completions API (vision‑capable model, default `gpt-4o-mini`).
6. The model replies with text; the server normalizes it to one of: `cat`, `dog`, `unknown`.
7. The UI displays the label and the raw response.

## Technology Used
- FastAPI: HTTP server and endpoints (`/classify_image`).
- Upload handling: `UploadFile` + `python-multipart` to parse multipart form data.
- OpenAI Python SDK (v1): `chat.completions.create` with mixed content (text + image).
- Prompting strategies:
  - Zero‑shot prompting: Instruction only. No examples.
  - Few‑shot prompting: Include 1–2 example image‑label pairs within the conversation to guide the model.
- Data URL encoding: The server embeds the image bytes as `data:<mime>;base64,<...>` so the model can read it directly.

## Why Prompting Instead of Training?
- Zero setup: No dataset curation, training loops, or model files.
- Flexible: Change instructions to handle more categories or stricter output without retraining.
- Good for demos and prototyping. For production, strict classes and SLAs may require a trained local model.

## Few‑Shot Prompting (How It Helps)
- The server can include:
  - A cat example (image → assistant: “cat”).
  - A dog example (image → assistant: “dog”).
- This narrows the model’s behavior, improving consistency and reducing ambiguous outputs.
- Still not training — it’s contextual guidance at inference time.

## Request Flow (Detailed)
- Frontend (static HTML/JS):
  - Gathers file(s) into `FormData`.
  - Optional: checkbox for few‑shot, and example images for cat/dog.
  - `fetch('/classify_image', { method: 'POST', body: formData })`.
- Backend (FastAPI):
  - Reads `file`, `few_shot`, `example_cat`, `example_dog`.
  - Converts images to data URLs, builds messages (system → optional examples → target image).
  - Calls OpenAI `chat.completions.create(model=CHAT_MODEL, messages=...)`.
  - Parses/normalizes response and returns JSON: `{ label, raw }`.

## Environment & Models
- Requires `OPENAI_API_KEY` with access/quota for a vision‑capable model.
- Defaults:
  - `CHAT_MODEL=gpt-4o-mini` (text + vision)
- You can override via environment variables.

## Limitations & Considerations
- Ambiguity: Poor lighting, occlusion, non‑standard pets may yield `unknown`.
- Cost/Quota: Each request calls a remote API; ensure billing/quota.
- Latency: Large images increase upload and processing time; consider resizing client‑side.
- Determinism: LLM outputs are probabilistic; we set `temperature=0` and enforce one‑word labels to stabilize results.

## Troubleshooting
- 401 Unauthorized / 429 Quota: Verify key and billing; test with a small image; confirm `CHAT_MODEL` access.
- “Form data requires python‑multipart”: Install `python-multipart` or run `pip install -r requirements.txt`.
- Vision access: Not all keys/orgs have access to all models; switch to an accessible model via `CHAT_MODEL`.
- Large files: Try a smaller JPG/PNG if requests are slow or time out.

## How To Demo (Script)
1. Start server: `python -m uvicorn app.main:app --reload`.
2. Open `http://127.0.0.1:8000`.
3. Basic: pick an image → “Classify”.
4. Few‑shot: tick “Use few‑shot examples”, upload a cat and/or dog example, then your target image → “Classify”.
5. Discuss results and show how few‑shot changes behavior without training.

## Extending to a Trained Local Model (Optional)
- Train a small CNN (MobileNetV2/ResNet50) on a cats‑vs‑dogs dataset.
- Export weights and serve with FastAPI.
- Trade‑offs: more setup and maintenance, but runs fully local and deterministic.
