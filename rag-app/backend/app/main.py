from fastapi import FastAPI
from app.schemas.users import UserIn, UserOut

app = FastAPI()

@app.post("/api/question")
async def ask_question(body: UserIn):
    
    user_query = body.question
    
    # chunks = get_chunks(user_query)
    # embedding = get_embedding(user_query)
    # docs = vector_search(embedding, k=body.top_k)
    # answer = llm.generate(user_query, docs)

    answer = "Ответ!"
    return UserOut(answer=answer)