from fastapi import FastAPI
from app.schemas.users import UserIn, UserOut
import os
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

app = FastAPI()

DATABASE_URL = os.getenv("DATABASE_URL")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)  # !!!
    return conn

def vector_search(query: str, top_k: int = 5):
    query_emb = model.encode([query])[0].tolist()

    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT content
        FROM documents
        ORDER BY embedding <-> %s::vector
        LIMIT %s
        """,
        (query_emb, top_k)
    )

    results = [row[0] for row in cur.fetchall()]

    cur.close()
    conn.close()

    return results

@app.post("/api/question")
async def ask_question(body: UserIn):
    user_query = body.question

    docs = vector_search(user_query, top_k=5)
    answer = "\n\n---\n\n".join(docs)

    return UserOut(answer=answer)