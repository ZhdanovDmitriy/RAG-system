import os
import psycopg2
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from app.schemas.users import UserIn, UserOut

DATABASE_URL = os.getenv("DATABASE_URL")
TOP_K_SHORT = 5
TOP_K_LONG = 5

app = FastAPI()

model = SentenceTransformer(
    "intfloat/multilingual-e5-base",
    cache_folder="/models/cache"
)

SYMBOL_MAP = {
    "unique existential quantifier": "∃!",
    "less than or equal to": "⩽",
    "greater than or equal to": "⩾",
    "logical equivalence": "⇐⇒",
    "logical consequence": "=⇒",
    "existential quantifier": "∃",
    "universal quantifier": "∀",
    "Cartesian product": "×",
    "set difference": "\\",       # Экранирование обратного слэша
    "element of": "∈",
    "subset of": "⊂",
    "empty set": "∅",
    "intersection": "∩",
    "union": "∪",
    "disjunction": "∨",
    "implication": "→",
    "conjunction": "&",           # Согласно моей таблице (обычно ∧)
    "negation": "¬",
    "inequality": "≠",            # Используем стандартный символ неравенства
    "equality": "=",
    "less than": "<",
    "greater than": ">",
    "bottom": "⊥",
    "turnstile": "⊢",
    "empty clause": "□",
    "angle bracket left": "⟨",
    "angle bracket right": "⟩",
    "composition": "◦",
    "conditionally equal": "≃",
    "equivalence": "∼",
    "ellipsis": "...",
}

# 2. Сортируем ключи по длине (убывание), чтобы сначала заменять длинные фразы
# Это предотвращает ситуацию, когда мы заменим "equal" внутри "less than or equal to"
SORTED_KEYS = sorted(SYMBOL_MAP.keys(), key=len, reverse=True)

def replace_symbols(text: str) -> str:
    """Заменяет английские названия на спецсимволы в тексте."""
    if not text:
        return text
    
    # Проходим по отсортированным ключам и делаем замену
    for key in SORTED_KEYS:
        # replace заменяет все вхождения ключа в строке
        text = text.replace(key, SYMBOL_MAP[key])
    
    return text

def get_conn():
    return psycopg2.connect(DATABASE_URL)

def embed_query(text: str):
    q = f"query: {text}"
    emb = model.encode(
        [q],
        normalize_embeddings=True,
    )[0]
    return emb.tolist()

def search_short(query, top_k):
    emb = embed_query(query)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT content,
               embedding <=> %s::vector AS distance
        FROM documents_short
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (emb, emb, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def search_long(query, top_k):
    emb = embed_query(query)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT content,
               embedding <=> %s::vector AS distance
        FROM documents_long
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (emb, emb, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

@app.post("/api/question", response_model=UserOut)
def search(q: UserIn):
    short_rows = search_short(q.question, 5)
    long_rows = search_long(q.question, 5)

    short = [replace_symbols(r[0]) for r in short_rows]
    long = [replace_symbols(r[0]) for r in long_rows]

    return UserOut(
        short=short,
        long=long,
    )