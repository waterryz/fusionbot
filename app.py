import os
import re
import json
import math
from typing import List, Dict, Any

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from openai import OpenAI

# ========= CONFIG =========
PDF_PATH = os.getenv("BROCHURE_PDF", "brochure.pdf")
CACHE_PATH = os.getenv("EMB_CACHE", "emb_cache.json")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # можно поменять
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Prime Fusion Brochure Bot")

# CORS: чтобы сайт мог дергать API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # лучше потом поставить твой домен
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= UTILS =========
def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return "\n".join(parts)

def clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    # режем по абзацам, чтобы смысл не ломать
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            # overlap: оставим хвост предыдущего
            if overlap > 0 and chunks:
                tail = chunks[-1][-overlap:]
                buf = (tail + "\n" + p).strip()
            else:
                buf = p
    if buf:
        chunks.append(buf)
    # уберем слишком короткие
    return [c for c in chunks if len(c) > 80]

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def embed_texts(texts: List[str]) -> List[List[float]]:
    # батчим одним запросом
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def safe_load_cache() -> Dict[str, Any]:
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_save_cache(data: Dict[str, Any]) -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

# ========= INDEX (in-memory) =========
INDEX_CHUNKS: List[str] = []
INDEX_EMB: np.ndarray | None = None

def build_or_load_index():
    global INDEX_CHUNKS, INDEX_EMB

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    raw = read_pdf_text(PDF_PATH)
    text = clean_text(raw)
    chunks = chunk_text(text)

    cache = safe_load_cache()
    # кеш валиден если совпадает кол-во чанков и сами чанки
    if cache.get("chunks") == chunks and "embeddings" in cache:
        INDEX_CHUNKS = chunks
        INDEX_EMB = np.array(cache["embeddings"], dtype=np.float32)
        return

    # строим embeddings и кешируем
    embeddings = embed_texts(chunks)
    safe_save_cache({"chunks": chunks, "embeddings": embeddings})

    INDEX_CHUNKS = chunks
    INDEX_EMB = np.array(embeddings, dtype=np.float32)

@app.on_event("startup")
def on_startup():
    build_or_load_index()

# ========= API =========
@app.get("/health")
def health():
    return {"ok": True, "chunks": len(INDEX_CHUNKS)}

@app.post("/chat")
def chat(payload: Dict[str, Any]):
    """
    payload:
      {
        "message": "текст вопроса",
        "history": [{"role":"user"|"assistant","content":"..."}]  (опционально)
      }
    """
    msg = (payload.get("message") or "").strip()
    if not msg:
        return {"answer": "Напиши вопрос — и я отвечу по брошюре."}

    if INDEX_EMB is None or not INDEX_CHUNKS:
        return {"answer": "Индекс не готов. Проверь PDF и перезапусти сервис."}

    # embedding вопроса
    q_emb = np.array(embed_texts([msg])[0], dtype=np.float32)

    # топ релевантных чанков
    scores = [cosine(q_emb, INDEX_EMB[i]) for i in range(len(INDEX_CHUNKS))]
    top_idx = np.argsort(scores)[-6:][::-1]  # top-6
    context = "\n\n---\n\n".join([INDEX_CHUNKS[i] for i in top_idx])

    system = (
        "Ты — помощник компании Prime Fusion Inc.\n"
        "Отвечай ТОЛЬКО по содержанию предоставленного документа (брошюры).\n"
        "Если в документе нет ответа — скажи: 'В брошюре нет этой информации' и предложи написать в WhatsApp/Telegram.\n"
        "Пиши по-русски. Термины TLC, AMT, WAV оставляй как есть. Uber/Lyft пиши как Убер/Лифт.\n"
        "Коротко, но понятно."
    )

    user_prompt = (
        "ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n"
        f"{msg}\n\n"
        "КОНТЕКСТ ИЗ БРОШЮРЫ:\n"
        f"{context}"
    )

    # можно подмешивать history если хочешь "память"
    history = payload.get("history") or []
    messages = [{"role": "system", "content": system}]
    # ограничим историю, чтобы не раздувать
    for h in history[-6:]:
        if isinstance(h, dict) and h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": str(h["content"])})
    messages.append({"role": "user", "content": user_prompt})

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=350,
    )

    answer = resp.choices[0].message.content.strip()
    return {"answer": answer}
