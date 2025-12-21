import os
import re
import json
from typing import List, Dict, Any

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from openai import OpenAI

# ========= CONFIG =========
PDF_PATH = os.getenv("BROCHURE_PDF", "brochure.pdf")
CACHE_PATH = os.getenv("EMB_CACHE", "emb_cache.json")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Prime Fusion Brochure Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= UTILS =========
def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join([(p.extract_text() or "") for p in reader.pages])

def clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def chunk_text(text: str, max_chars=1200, overlap=200) -> List[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, buf = [], ""

    for p in paras:
        if len(buf) + len(p) <= max_chars:
            buf = (buf + "\n" + p).strip()
        else:
            chunks.append(buf)
            buf = (buf[-overlap:] + "\n" + p).strip() if overlap else p

    if buf:
        chunks.append(buf)

    return [c for c in chunks if len(c) > 80]

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0

def embed_texts(texts: List[str]) -> List[List[float]]:
    r = client.embeddings.create(model=EMB_MODEL, input=texts)
    return [x.embedding for x in r.data]

def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(data):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

# ========= INDEX =========
INDEX_CHUNKS: List[str] = []
INDEX_EMB: np.ndarray | None = None

def build_index():
    global INDEX_CHUNKS, INDEX_EMB

    raw = read_pdf_text(PDF_PATH)
    text = clean_text(raw)
    chunks = chunk_text(text)

    cache = load_cache()
    if cache.get("chunks") == chunks:
        INDEX_CHUNKS = chunks
        INDEX_EMB = np.array(cache["embeddings"], dtype=np.float32)
        return

    emb = embed_texts(chunks)
    save_cache({"chunks": chunks, "embeddings": emb})

    INDEX_CHUNKS = chunks
    INDEX_EMB = np.array(emb, dtype=np.float32)

@app.on_event("startup")
def startup():
    build_index()

# ========= API =========
@app.get("/health")
def health():
    return {"ok": True, "chunks": len(INDEX_CHUNKS)}

@app.post("/chat")
def chat(payload: Dict[str, Any]):
    msg = (payload.get("message") or "").strip()
    if not msg:
        return {"answer": "Please enter a question."}

    lang = payload.get("lang", "ru")

    q_emb = np.array(embed_texts([msg])[0], dtype=np.float32)
    scores = [cosine(q_emb, INDEX_EMB[i]) for i in range(len(INDEX_CHUNKS))]
    top = np.argsort(scores)[-6:][::-1]

    context = "\n\n---\n\n".join([INDEX_CHUNKS[i] for i in top])

    if lang == "ru":
        system = (
            "Ты — ИИ-помощник компании Prime Fusion Inc.\n"
            "Отвечай ТОЛЬКО на основе текста брошюры.\n"
            "Если информации нет — скажи, что её нет в брошюре, и предложи связаться через WhatsApp или Telegram.\n"
            "Пиши по-русски. Термины TLC, AMT, WAV не переводить.\n"
            "Uber и Lyft пиши как Убер и Лифт.\n"
            "Отвечай кратко и по делу."
        )
    else:
        system = (
            "You are an AI assistant for Prime Fusion Inc.\n"
            "Answer ONLY using the provided brochure text.\n"
            "If the information is not available, clearly state that and suggest contacting via WhatsApp or Telegram.\n"
            "Keep the answer short and professional."
        )

    user_prompt = f"""
USER QUESTION:
{msg}

BROCHURE CONTEXT:
{context}
"""

    messages = [{"role": "system", "content": system}]
    for h in (payload.get("history") or [])[-6:]:
        if h.get("role") in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})

    messages.append({"role": "user", "content": user_prompt})

    r = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=350
    )

    return {"answer": r.choices[0].message.content.strip()}
