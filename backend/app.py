from fastapi import FastAPI
from pydantic import BaseModel
from src.retriever import Retriever
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
retriever = Retriever()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class Query(BaseModel):
    question: str

@app.post("/query")
async def query(q: Query):
    passages = retriever.query(q.question, top_k=3)
    answer = passages[0] if passages else "Bilgi bulunamadı."
    return {"answer": answer, "passages": passages}
