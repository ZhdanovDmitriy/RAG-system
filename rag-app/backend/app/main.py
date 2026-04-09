import os
import json
import random
import psycopg2
import numpy as np
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from app.schemas.users import UserIn, UserOut, TestResponse, TestQuestion, TestSubmission, TestResult, TestResultItem

DATABASE_URL = os.getenv("DATABASE_URL")
SIMILARITY_THRESHOLD = 0.85  # порог косинусного сходства
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


# ========================
#  TEST ENDPOINTS
# ========================


def load_test_questions_from_db():
    """Загружает вопросы из БД."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, question_text FROM test_questions ORDER BY id")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"id": r[0], "question": r[1]} for r in rows]


@app.get("/api/test/generate", response_model=TestResponse)
def generate_test():
    questions = load_test_questions_from_db()
    # Случайные 5 из всех
    selected = random.sample(questions, k=5)
    return TestResponse(
        questions=[
            TestQuestion(id=q["id"], question=q["question"])
            for q in selected
        ]
    )


def embed_text(text: str):
    emb = model.encode(
        [text],
        normalize_embeddings=True,
    )[0]
    return emb.tolist()


def find_best_match_for_question(question_id: int, user_emb):
    """Ищем лучший ответ для данного вопроса среди эталонных эмбеддингов."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT answer_text,
               1 - (embedding <=> %s::vector) AS similarity
        FROM test_answer_embeddings
        WHERE question_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT 1
        """,
        (user_emb, question_id, user_emb),
    )

    row = cur.fetchone()
    cur.close()
    conn.close()

    if row is None:
        return None, 0.0

    return row[0], row[1]


@app.post("/api/test/submit", response_model=TestResult)
def submit_test(submission: TestSubmission):
    questions = load_test_questions_from_db()
    q_map = {q["id"]: q for q in questions}

    results = []
    score = 0

    for ans in submission.answers:
        q_id = ans.question_id
        user_answer = ans.answer.strip()

        question_text = q_map[q_id]["question"]

        if not user_answer:
            # Пустой ответ — автоматически неправильный
            best_answer, _ = find_best_match_for_question(q_id, [0.0] * 768)
            results.append(
                TestResultItem(
                    question_id=q_id,
                    question=question_text,
                    is_correct=False,
                    similarity=0.0,
                    best_correct_answer=best_answer,
                )
            )
            continue

        user_emb = embed_text(f"query: {user_answer}")

        best_answer, similarity = find_best_match_for_question(
            q_id, user_emb
        )

        # similarity = 1 - cosine_distance, т.к. embeddings нормализованы
        # Это и есть косинусное сходство
        is_correct = similarity >= SIMILARITY_THRESHOLD

        if is_correct:
            score += 1

        results.append(
            TestResultItem(
                question_id=q_id,
                question=question_text,
                is_correct=is_correct,
                similarity=round(similarity, 4),
                best_correct_answer=best_answer,
            )
        )

    # Оценка
    if score <= 2:
        grade = "Неудовлетворительно"
    elif score == 3:
        grade = "Удовлетворительно"
    elif score == 4:
        grade = "Хорошо"
    else:
        grade = "Отлично"

    return TestResult(
        score=score,
        total=5,
        grade=grade,
        results=results,
    )