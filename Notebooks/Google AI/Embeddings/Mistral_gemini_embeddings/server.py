"""
server.py — RAG Multimodal Chat API
────────────────────────────────────
FastAPI + SSE streaming + ChromaDB + Gemini Embeddings

Requiere (generados por el notebook):
  ./chroma_db_mistral_gemini_vision/
  ./image_manifest.json
  ./extracted_images/

Uso:
  pip install fastapi uvicorn[standard]
  export GEMINI_API_KEY="..."
  python server.py
  → http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import AsyncGenerator

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

CHROMA_PATH     = "./chroma_db_mistral_gemini_vision"
COLLECTION_NAME = "multimodal_rag"
EMBED_MODEL     = "gemini-embedding-2-preview"
GEN_MODEL       = "gemini-3-flash-preview"
MANIFEST_PATH   = "./data/image_manifest.json"
IMAGES_DIR      = "./data/extracted_images"
MAX_HISTORY     = 8   # mensajes del historial incluidos en el prompt

# ══════════════════════════════════════════════════════════════
# STARTUP — cargar clientes y datos
# ══════════════════════════════════════════════════════════════

api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    raise EnvironmentError(
        "❌  GEMINI_API_KEY no encontrada.\n"
        "   Configúrala: export GEMINI_API_KEY='tu_clave'"
    )

gemini = genai.Client(api_key=api_key)

chroma = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    col = chroma.get_collection(name=COLLECTION_NAME)
except Exception:
    raise RuntimeError(
        f"❌  No se encontró la colección '{COLLECTION_NAME}'.\n"
        "   Ejecuta el notebook primero."
    )

if not Path(MANIFEST_PATH).exists():
    raise FileNotFoundError(
        f"❌  No se encontró '{MANIFEST_PATH}'.\n"
        "   Ejecuta la celda de exportación del notebook."
    )

with open(MANIFEST_PATH, encoding="utf-8") as fh:
    _manifest = json.load(fh)

images: dict[str, dict] = {item["id"]: item for item in _manifest}

n_images = len(images)
n_text   = col.count() - n_images

print(f"✅  Colección '{COLLECTION_NAME}': {col.count()} ítems")
print(f"    📝 {n_text} chunks de texto  |  🖼️  {n_images} imágenes")

# ══════════════════════════════════════════════════════════════
# RAG CORE
# ══════════════════════════════════════════════════════════════

def _embed(query: str) -> list[float]:
    r = gemini.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return r.embeddings[0].values


def dual_retrieve(query: str, k_text: int, k_images: int):
    qv  = _embed(query)
    k_t = min(k_text,   max(1, n_text))
    k_i = min(k_images, max(0, n_images))

    raw_t = col.query(query_embeddings=[qv], n_results=k_t, where={"type": "text"})
    raw_i = col.query(query_embeddings=[qv], n_results=k_i, where={"type": "image"}) if k_i > 0 else {"ids": [[]], "documents": [[]], "distances": [[]]}

    text_hits = [
        {"id": did, "doc": doc, "sim": round(1.0 - dist, 4)}
        for did, doc, dist in zip(raw_t["ids"][0], raw_t["documents"][0], raw_t["distances"][0])
    ]
    img_hits = [
        {"id": did, "sim": round(1.0 - dist, 4)}
        for did, dist in zip(raw_i["ids"][0], raw_i["distances"][0])
        if did in images
    ]
    return text_hits, img_hits


def build_prompt(question: str, text_hits: list, img_hits: list, history: list) -> str:
    text_ctx = "\n\n".join(f"[{r['id']} sim={r['sim']}]\n{r['doc']}" for r in text_hits)
    img_ctx  = "\n".join(
        f"[{h['id']} sim={h['sim']}] {images[h['id']]['alt']}"
        for h in img_hits
    ) or "Sin imágenes recuperadas."

    hist_lines = []
    for msg in history[-(MAX_HISTORY * 2):]:
        role    = "Usuario" if msg.get("role") == "user" else "Asistente"
        content = msg.get("content", "")[:600]
        if content:
            hist_lines.append(f"{role}: {content}")

    hist_block = f"<HISTORIAL>\n{chr(10).join(hist_lines)}\n</HISTORIAL>\n\n" if hist_lines else ""

    return (
        "Eres un asistente experto en documentos técnicos. "
        "Responde con precisión y claridad usando el contexto proporcionado. "
        "Cuando corresponda, menciona explícitamente las imágenes adjuntas que apoyan tu respuesta.\n\n"
        f"{hist_block}"
        f"<PREGUNTA>\n{question}\n</PREGUNTA>\n\n"
        f"<CONTEXTO_TEXTO>\n{text_ctx}\n</CONTEXTO_TEXTO>\n\n"
        f"<CONTEXTO_IMAGENES>\n{img_ctx}\n</CONTEXTO_IMAGENES>"
    )


# ══════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(title="RAG Multimodal Chat", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Imágenes extraídas como archivos estáticos → /images/img-0.jpeg
if Path(IMAGES_DIR).is_dir():
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


# ── Rutas ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def ui():
    html = Path("index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/info")
async def info():
    return {
        "collection": COLLECTION_NAME,
        "total":      col.count(),
        "n_text":     n_text,
        "n_images":   n_images,
    }


class ChatRequest(BaseModel):
    message:    str
    history:    list  = []
    k_text:     int   = 4
    k_images:   int   = 4
    show_sources: bool = False


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Endpoint SSE:  text/event-stream
    Eventos emitidos (JSON):
      {type: "images",  data: [{id, url, sim, alt}, ...]}
      {type: "sources", data: [{id, doc, sim},      ...]}
      {type: "text",    content: "..."}    ← fragmentos de streaming
      {type: "done"}
      {type: "error",   message: "..."}
    """
    loop  = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run():
        """Ejecuta retrieval + streaming de Gemini en un thread."""
        try:
            text_hits, img_hits = dual_retrieve(req.message, req.k_text, req.k_images)

            # ── Imágenes ──
            if img_hits:
                payload = [
                    {
                        "id":  h["id"],
                        "url": f"/images/{h['id']}",
                        "sim": h["sim"],
                        "alt": images[h["id"]]["alt"],
                    }
                    for h in img_hits
                ]
                asyncio.run_coroutine_threadsafe(queue.put(("images", payload)), loop)

            # ── Fuentes de texto ──
            if req.show_sources and text_hits:
                asyncio.run_coroutine_threadsafe(queue.put(("sources", text_hits)), loop)

            # ── Construir contenidos multimodales ──
            prompt = build_prompt(req.message, text_hits, img_hits, req.history)
            parts  = [prompt]
            for h in img_hits:
                p = images[h["id"]]["path"]
                if Path(p).exists():
                    parts.append(
                        types.Part.from_bytes(
                            data=Path(p).read_bytes(),
                            mime_type=images[h["id"]]["mime_type"],
                        )
                    )

            # ── Stream Gemini ──
            for chunk in gemini.models.generate_content_stream(
                model=GEN_MODEL, contents=parts
            ):
                if chunk.text:
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("text", chunk.text)), loop
                    )

        except Exception as exc:
            asyncio.run_coroutine_threadsafe(queue.put(("error", str(exc))), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)

    threading.Thread(target=_run, daemon=True).start()

    async def _generate() -> AsyncGenerator[str, None]:
        while True:
            kind, data = await queue.get()
            if kind == "done":
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                break
            elif kind == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': data})}\n\n"
                break
            elif kind == "text":
                yield f"data: {json.dumps({'type': 'text', 'content': data})}\n\n"
            elif kind == "images":
                yield f"data: {json.dumps({'type': 'images', 'data': data})}\n\n"
            elif kind == "sources":
                yield f"data: {json.dumps({'type': 'sources', 'data': data})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    print("\n🚀  RAG Multimodal Chat")
    print("    http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="warning")
