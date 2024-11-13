from fastapi import FastAPI
from pydantic import BaseModel
from src.chain import handle_user_input as med_QA 
from typing import List, Any, Optional


app = FastAPI(
    title="RAG_APP",
    description="Retrieval Augmented Generation APP which lets users ask questions and get answers using LLMs",
)

class UserQuestion(BaseModel):
    question: str

class UserAnswer(UserQuestion):
    question: Optional[str]
    answer: Optional[str] = None
    chat_history: Optional[List[Any]] = None


@app.get("/")
def index():
    return {"message": "Hello World"}

@app.post("/questions",response_model=UserAnswer)
def question(user_message: UserQuestion):
    answer,his = med_QA(user_message.question)
    bot_response= UserAnswer(question=user_message.question,answer=answer,chat_history=his)
    return bot_response

