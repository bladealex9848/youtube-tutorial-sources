"""
chat_rag_multimodal.py
──────────────────────
Interfaz web de chat sobre documentos PDF procesados con Mistral OCR 3
y embeddings multimodales de Gemini.

Requiere (generados por el notebook Tutorial_Mistral_Gemini_RAG.ipynb):
  - ./chroma_db_mistral_gemini_vision/   ChromaDB con embeddings y metadata
  - ./image_manifest.json                Lista de imágenes con paths en disco
  - ./extracted_images/                  Archivos de imagen exportados

Variables de entorno:
  GEMINI_API_KEY   (obligatorio)

Uso:
  pip install gradio
  python chat_rag_multimodal.py
"""

import os
import json
import base64
import textwrap

import gradio as gr
import chromadb
from google import genai
from google.genai import types

# ═══════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════

CHROMA_PATH      = "./chroma_db_mistral_gemini_vision"
COLLECTION_NAME  = "multimodal_rag"
EMBEDDING_MODEL  = "gemini-embedding-2-preview"
GENERATION_MODEL = "gemini-2.5-flash-preview-05-20"
MANIFEST_PATH    = "./data/image_manifest.json"
MAX_HISTORY      = 6   # mensajes del historial enviados a Gemini (pares user/assistant)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "❌ GEMINI_API_KEY no encontrada.\n"
        "   Configúrala con: export GEMINI_API_KEY='tu_clave'"
    )

# ═══════════════════════════════════════════════════════════
# 2. INICIALIZAR CLIENTES Y DATOS
# ═══════════════════════════════════════════════════════════

print("⏳ Cargando recursos...")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection    = chroma_client.get_collection(name=COLLECTION_NAME)

if not os.path.exists(MANIFEST_PATH):
    raise FileNotFoundError(
        f"❌ No se encontró '{MANIFEST_PATH}'.\n"
        "   Ejecuta la celda de exportación del notebook primero."
    )

with open(MANIFEST_PATH, encoding="utf-8") as f:
    manifest_list = json.load(f)

images_dict = {item["id"]: item for item in manifest_list}

n_text   = collection.count() - len(images_dict)
n_images = len(images_dict)

print(f"✅ Colección '{COLLECTION_NAME}': {collection.count()} ítems")
print(f"   📝 {n_text} chunks de texto  |  🖼️  {n_images} imágenes")
print("✅ Listo.\n")


# ═══════════════════════════════════════════════════════════
# 3. NÚCLEO RAG
# ═══════════════════════════════════════════════════════════

def embed_query(query: str) -> list[float]:
    """Genera el embedding de una query de usuario."""
    resp = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return resp.embeddings[0].values


def dual_retrieve(query: str, k_text: int, k_images: int) -> tuple[list, list]:
    """
    Consulta texto e imágenes en pools separados para garantizar
    representación balanceada independientemente de las similitudes.

    Returns:
        text_hits:  lista de {id, doc, sim}
        image_hits: lista de {id, doc, sim, img_meta}
    """
    qv = embed_query(query)

    k_t = min(k_text,  max(1, n_text))
    k_i = min(k_images, n_images)

    raw_text  = collection.query(query_embeddings=[qv], n_results=k_t,  where={"type": "text"})
    raw_image = collection.query(query_embeddings=[qv], n_results=k_i,  where={"type": "image"})

    text_hits = [
        {"id": did, "doc": doc, "sim": round(1.0 - dist, 4)}
        for did, doc, dist in zip(
            raw_text["ids"][0], raw_text["documents"][0], raw_text["distances"][0]
        )
    ]
    image_hits = [
        {"id": did, "doc": doc, "sim": round(1.0 - dist, 4), "img": images_dict[did]}
        for did, doc, dist in zip(
            raw_image["ids"][0], raw_image["documents"][0], raw_image["distances"][0]
        )
        if did in images_dict
    ]

    return text_hits, image_hits


def build_prompt(question: str, text_hits: list, image_hits: list, history: list) -> str:
    """Construye el prompt de texto para Gemini con contexto RAG e historial."""
    text_ctx = "\n\n".join(
        f"[{r['id']} | sim={r['sim']}]\n{r['doc']}"
        for r in text_hits
    )
    img_ctx = "\n".join(
        f"[{r['id']} | sim={r['sim']}] {r['img']['alt']}"
        for r in image_hits
    ) or "Ninguna imagen recuperada."

    # Últimos N mensajes del historial (formato plano)
    hist_lines = []
    for msg in history[-(MAX_HISTORY * 2):]:
        role = "Usuario" if msg["role"] == "user" else "Asistente"
        # Strip HTML del historial para no contaminar el contexto
        clean = msg["content"].split("<")[0].strip()
        if clean:
            hist_lines.append(f"{role}: {clean}")
    hist_str = "\n".join(hist_lines)

    return textwrap.dedent(f"""
        Eres un asistente experto en documentos técnicos. Responde con precisión y claridad.
        Cuando corresponda, menciona explícitamente las imágenes que apoyan tu respuesta.

        {f"<HISTORIAL_CONVERSACION>{hist_str}</HISTORIAL_CONVERSACION>" if hist_str else ""}

        <PREGUNTA>{question}</PREGUNTA>

        <CONTEXTO_TEXTO>
        {text_ctx}
        </CONTEXTO_TEXTO>

        <CONTEXTO_IMAGENES>
        {img_ctx}
        </CONTEXTO_IMAGENES>
    """).strip()


# ═══════════════════════════════════════════════════════════
# 4. HELPERS DE FORMATO HTML
# ═══════════════════════════════════════════════════════════

def _img_to_b64(path: str, mime: str) -> str:
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def render_image_card(hit: dict) -> str:
    """Genera un card HTML con imagen, similitud y descripción corta."""
    img  = hit["img"]
    src  = _img_to_b64(img["path"], img["mime_type"])
    desc = img["alt"][:110] + "…" if len(img["alt"]) > 110 else img["alt"]
    sim_pct = int(hit["sim"] * 100)
    bar_color = "#22c55e" if sim_pct >= 70 else "#f59e0b" if sim_pct >= 50 else "#94a3b8"
    return (
        f'<div style="display:inline-flex;flex-direction:column;'
        f'width:220px;margin:6px;border:1px solid #e2e8f0;'
        f'border-radius:10px;overflow:hidden;background:#fff;'
        f'box-shadow:0 1px 4px rgba(0,0,0,.08);vertical-align:top;">'
        f'<img src="{src}" style="width:220px;height:160px;object-fit:cover;" />'
        f'<div style="padding:8px 10px;">'
        f'<div style="font-size:11px;font-weight:600;color:#475569;margin-bottom:3px;">'
        f'{img["id"]}</div>'
        f'<div style="height:4px;border-radius:2px;background:#e2e8f0;margin-bottom:5px;">'
        f'<div style="height:4px;width:{sim_pct}%;border-radius:2px;background:{bar_color};"></div></div>'
        f'<div style="font-size:10px;color:#64748b;line-height:1.35;">{desc}</div>'
        f'</div></div>'
    )


def render_sources_detail(text_hits: list) -> str:
    """Genera el bloque <details> de fuentes de texto."""
    rows = "".join(
        f'<div style="margin-bottom:10px;padding:8px;background:#f8fafc;'
        f'border-left:3px solid #6366f1;border-radius:4px;">'
        f'<span style="font-size:11px;font-weight:700;color:#4f46e5;">{r["id"]}</span>'
        f'<span style="font-size:11px;color:#94a3b8;margin-left:8px;">sim {r["sim"]}</span>'
        f'<p style="font-size:12px;color:#475569;margin:4px 0 0;">{r["doc"][:280]}…</p>'
        f'</div>'
        for r in text_hits
    )
    return (
        f'<details style="margin-top:14px;">'
        f'<summary style="cursor:pointer;font-size:12px;color:#94a3b8;'
        f'user-select:none;">🔍 Ver {len(text_hits)} fuentes de texto</summary>'
        f'<div style="margin-top:8px;">{rows}</div>'
        f'</details>'
    )


# ═══════════════════════════════════════════════════════════
# 5. FUNCIÓN PRINCIPAL DE CHAT (streaming)
# ═══════════════════════════════════════════════════════════

def respond(
    message: str,
    history: list,
    k_text: int,
    k_images: int,
    show_sources: bool,
):
    """
    Genera la respuesta del asistente con streaming.

    El historial usa el formato de mensajes de Gradio:
      [{"role": "user"/"assistant", "content": str}, ...]
    """
    if not message.strip():
        yield history
        return

    # ── Añadir mensaje del usuario al historial ──
    history = history + [{"role": "user", "content": message}]
    yield history

    # ── Dual retrieval ──
    try:
        text_hits, image_hits = dual_retrieve(message, k_text=k_text, k_images=k_images)
    except Exception as exc:
        history = history + [{"role": "assistant", "content": f"⚠️ Error en recuperación: {exc}"}]
        yield history
        return

    # ── Construir prompt y partes multimodales ──
    prompt = build_prompt(message, text_hits, image_hits, history[:-1])

    # Imágenes como partes binarias para Gemini
    image_parts = []
    for hit in image_hits:
        img = hit["img"]
        if os.path.exists(img["path"]):
            with open(img["path"], "rb") as f:
                image_parts.append(
                    types.Part.from_bytes(data=f.read(), mime_type=img["mime_type"])
                )

    contents = [prompt] + image_parts

    # ── Streaming de la respuesta ──
    partial = ""
    try:
        for chunk in gemini_client.models.generate_content_stream(
            model=GENERATION_MODEL,
            contents=contents,
        ):
            if chunk.text:
                partial += chunk.text
                history_stream = history + [{"role": "assistant", "content": partial}]
                yield history_stream
    except Exception as exc:
        history = history + [{"role": "assistant", "content": f"⚠️ Error en generación: {exc}"}]
        yield history
        return

    # ── Adjuntar imágenes recuperadas ──
    full_content = partial

    if image_hits:
        cards = "".join(render_image_card(h) for h in image_hits)
        full_content += (
            '\n\n<div style="border-top:1px solid #e2e8f0;margin-top:14px;padding-top:12px;">'
            '<p style="font-size:12px;font-weight:600;color:#64748b;margin:0 0 8px;">'
            '📷 Imágenes recuperadas del documento</p>'
            f'<div style="display:flex;flex-wrap:wrap;gap:4px;">{cards}</div>'
            '</div>'
        )

    if show_sources and text_hits:
        full_content += render_sources_detail(text_hits)

    history = history + [{"role": "assistant", "content": full_content}]
    yield history


# ═══════════════════════════════════════════════════════════
# 6. INTERFAZ GRADIO
# ═══════════════════════════════════════════════════════════

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
)

CSS = """
#chatbot { border: none !important; }
#chatbot .message.bot { background: #f8fafc !important; }
footer { display: none !important; }
.contain { max-width: 1200px !important; }
"""

EXAMPLE_QUESTIONS = [
    "¿Cómo se realiza el proceso de instalación paso a paso?",
    "¿Qué componentes tiene el equipo y para qué sirve cada uno?",
    "¿Cuáles son las instrucciones de seguridad que debo seguir?",
    "Describe los diagramas o esquemas técnicos del documento",
    "¿Cómo se realiza el mantenimiento del equipo?",
]

with gr.Blocks(theme=THEME, css=CSS, title="RAG Multimodal Chat") as demo:

    # ── Header ──
    gr.HTML("""
        <div style="display:flex;align-items:center;gap:12px;
                    padding:18px 0 10px;border-bottom:1px solid #e2e8f0;margin-bottom:16px;">
          <div style="font-size:32px;">💬</div>
          <div>
            <h1 style="margin:0;font-size:22px;font-weight:700;color:#1e293b;">
              RAG Multimodal Chat</h1>
            <p style="margin:2px 0 0;font-size:13px;color:#64748b;">
              Mistral OCR 3 · Gemini Embeddings · Dual Retrieval</p>
          </div>
        </div>
    """)

    with gr.Row(equal_height=True):

        # ── Panel lateral ──
        with gr.Column(scale=1, min_width=240):

            gr.Markdown("#### ⚙️ Parámetros de recuperación")

            k_text_slider = gr.Slider(
                minimum=1, maximum=10, value=4, step=1,
                label="Chunks de texto (k_text)",
                info="Fragmentos de texto por consulta",
            )
            k_images_slider = gr.Slider(
                minimum=0, maximum=8, value=4, step=1,
                label="Imágenes (k_images)",
                info="Imágenes recuperadas por consulta",
            )
            show_sources_cb = gr.Checkbox(
                value=False,
                label="Mostrar fuentes de texto",
            )

            gr.Markdown("---")
            gr.HTML(f"""
                <div style="font-size:12px;color:#64748b;line-height:1.7;">
                  <b>Índice activo</b><br/>
                  📦 <code>{COLLECTION_NAME}</code><br/>
                  🗂️ {collection.count()} ítems totales<br/>
                  📝 {n_text} chunks de texto<br/>
                  🖼️ {n_images} imágenes indexadas
                </div>
            """)

            gr.Markdown("---")
            gr.Markdown("#### 💡 Preguntas de ejemplo")
            example_btns = [
                gr.Button(q, variant="secondary", size="sm")
                for q in EXAMPLE_QUESTIONS
            ]

        # ── Área de chat ──
        with gr.Column(scale=4):

            chatbot = gr.Chatbot(
                value=[],
                type="messages",
                height=560,
                show_label=False,
                elem_id="chatbot",
                render_markdown=True,
                bubble_full_width=False,
                avatar_images=(
                    None,
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Google_Gemini_logo.svg/64px-Google_Gemini_logo.svg.png",
                ),
            )

            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Escribe tu pregunta sobre el documento...",
                    show_label=False,
                    scale=9,
                    lines=1,
                    max_lines=4,
                    autofocus=True,
                    submit_btn=False,
                )
                send_btn = gr.Button("Enviar ↩", variant="primary", scale=1, min_width=90)

            with gr.Row():
                clear_btn = gr.Button("🗑️ Limpiar chat", size="sm", variant="secondary")
                gr.HTML(
                    '<div style="font-size:11px;color:#94a3b8;padding-top:6px;">'
                    "Las respuestas se generan con el contexto de los últimos "
                    f"{MAX_HISTORY} turnos de conversación.</div>"
                )

    # ── Estado (historial interno) ──
    state = gr.State([])

    # ── Wiring de eventos ──
    def _submit(message, history, k_text, k_images, show_src):
        """Wrapper: hace yield de los estados parciales para streaming."""
        for updated_history in respond(message, history, k_text, k_images, show_src):
            yield updated_history, updated_history, ""

    submit_inputs  = [msg_box, state, k_text_slider, k_images_slider, show_sources_cb]
    submit_outputs = [chatbot, state, msg_box]

    send_btn.click(
        fn=_submit,
        inputs=submit_inputs,
        outputs=submit_outputs,
    )
    msg_box.submit(
        fn=_submit,
        inputs=submit_inputs,
        outputs=submit_outputs,
    )

    def clear_chat():
        return [], []

    clear_btn.click(fn=clear_chat, outputs=[chatbot, state])

    # Botones de ejemplo: llenan el textbox
    for btn, q in zip(example_btns, EXAMPLE_QUESTIONS):
        btn.click(fn=lambda text=q: text, outputs=msg_box)


# ═══════════════════════════════════════════════════════════
# 7. ARRANQUE
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
    )
