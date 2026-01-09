import os
import re
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from openai import OpenAI

# ========= CONFIG =========

PDF_PATH = os.getenv("BROCHURE_PDF", "brochure.pdf")
CACHE_PATH = os.getenv("EMB_CACHE", "emb_cache.json")
FAQ_PATH = os.getenv("FAQ_JSON", "faq.json")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small")

# 🔒 Railway volume (ВАЖНО)
CHAT_LOG_PATH = os.getenv("CHAT_LOG_PATH", "/data/chat_logs.json")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Prime Fusion Brochure Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= CHAT LOG STORAGE =========

def load_chat_logs():
    if os.path.exists(CHAT_LOG_PATH):
        with open(CHAT_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_chat_logs(logs):
    os.makedirs(os.path.dirname(CHAT_LOG_PATH), exist_ok=True)
    with open(CHAT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

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

# ========= FAQ =========

FAQ_DATA = {"ru": [], "en": []}
FAQ_EMB = {"ru": None, "en": None}

def load_faq():
    if not os.path.exists(FAQ_PATH):
        return

    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for lang in ("ru", "en"):
        FAQ_DATA[lang] = data.get(lang, [])
        if FAQ_DATA[lang]:
            questions = [x["q"] for x in FAQ_DATA[lang]]
            FAQ_EMB[lang] = np.array(embed_texts(questions), dtype=np.float32)

def normalize(t: str) -> str:
    return re.sub(r"[^\w\s]", "", t.lower()).strip()


def find_faq_answer(lang: str, question: str, threshold: float = 0.7):
    if not FAQ_DATA.get(lang):
        return None

    q_norm = normalize(question)

    # 1️⃣ FAST MATCH (exact / keyword)
    for item in FAQ_DATA[lang]:
        if normalize(item["q"]) in q_norm or q_norm in normalize(item["q"]):
            return item["a"]

    # 2️⃣ EMBEDDING MATCH
    if FAQ_EMB.get(lang) is None:
        return None

    q_emb = np.array(embed_texts([question])[0], dtype=np.float32)
    scores = [cosine(q_emb, FAQ_EMB[lang][i]) for i in range(len(FAQ_DATA[lang]))]
    idx = int(np.argmax(scores))

    if scores[idx] >= threshold:
        return FAQ_DATA[lang][idx]["a"]

    return None


# ========= INDEX =========

INDEX_CHUNKS: List[str] = []
INDEX_EMB: np.ndarray | None = None


def build_index():
    global INDEX_CHUNKS, INDEX_EMB

    texts = []

    # ===== BROCHURE (PDF) =====
    if os.path.exists(PDF_PATH):
        raw = read_pdf_text(PDF_PATH)
        texts.append(clean_text(raw))

    # ===== AGREEMENT / POLICIES (TXT) =====
    extra_files = [
        "rentalRU (16).txt",
        "policies.txt",
        "terms.txt",
    ]

    for path in extra_files:
        if not os.path.exists(path):
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()

            texts.append(clean_text(raw))

        except Exception as e:
            print(f"[INDEX] Failed to read {path}: {e}")

    if not texts:
        INDEX_CHUNKS = []
        INDEX_EMB = None
        return

    # ===== MERGE & CHUNK =====
    full_text = "\n\n".join(texts)
    chunks = chunk_text(full_text)

    # ===== CACHE CHECK =====
    cache = load_cache()
    if cache.get("chunks") == chunks:
        INDEX_CHUNKS = chunks
        INDEX_EMB = np.array(cache["embeddings"], dtype=np.float32)
        return

    # ===== EMBEDDINGS =====
    emb = embed_texts(chunks)
    save_cache({
        "chunks": chunks,
        "embeddings": emb
    })

    INDEX_CHUNKS = chunks
    INDEX_EMB = np.array(emb, dtype=np.float32)


@app.on_event("startup")
def startup():
    load_faq()
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

    # FAQ priority
    faq_answer = find_faq_answer(lang, msg)
    if faq_answer:
        return {"answer": faq_answer}

    q_emb = np.array(embed_texts([msg])[0], dtype=np.float32)
    scores = [cosine(q_emb, INDEX_EMB[i]) for i in range(len(INDEX_CHUNKS))]
    top = np.argsort(scores)[-6:][::-1]

    context = "\n\n---\n\n".join([INDEX_CHUNKS[i] for i in top])

    system = (
        "Ты — виртуальный помощник компании Prime Fusion Inc.\n"
        "Ты НЕ называешь имён людей и НЕ представляешься человеком.\n"
        "Отвечай на основе:\n"
        "1) FAQ\n"
        "2) Договора аренды и внутренних правил\n"
        "3) Брошюры\n\n"
        "Если точного ответа нет:\n"
        "- дай ОБЩУЮ информацию, если это безопасно\n"
        "- либо задай ОДИН уточняющий вопрос\n"
        "- либо предложи связаться через email,Telegram Bot\n\n"
        "НЕ используй фразы: «в брошюре нет информации».\n"
        "Отвечай уверенно, кратко и по-делу."
        if lang == "ru" else
        "You are a virtual assistant for Prime Fusion Inc.\n"
        "You do NOT use personal names and do NOT claim to be human.\n"
        "Answer based on:\n"
        "1) FAQ\n"
        "2) Rental agreement and internal policies\n"
        "3) Brochure\n\n"
        "If no exact answer exists:\n"
        "- give general guidance if safe\n"
        "- or ask ONE clarifying question\n"
        "- or suggest contacting via email or Telegram Bot\n\n"
        "Do NOT say: 'this information is not in the brochure'."
    )


    messages = [{"role": "system", "content": system}]
    for h in (payload.get("history") or [])[-6:]:
        if h.get("role") in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})

    messages.append({
        "role": "user",
        "content": f"USER QUESTION:\n{msg}\n\nBROCHURE CONTEXT:\n{context}"
    })

    r = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=350
    )

    answer = r.choices[0].message.content.strip()

    # ===== SAVE CHAT (NO SIDE EFFECTS) =====
    try:
        logs = load_chat_logs()
        logs.append({
            "id": str(uuid.uuid4()),
            "ts": datetime.utcnow().isoformat(),
            "lang": lang,
            "question": msg,
            "answer": answer
        })
        save_chat_logs(logs)
    except Exception:
        pass  # 🔒 никогда не ломаем чат

    return {"answer": answer}

# ========= ADMIN READ =========

@app.get("/admin/ai-chats")
def get_ai_chats():
    return load_chat_logs()[::-1]  # новые сверху





