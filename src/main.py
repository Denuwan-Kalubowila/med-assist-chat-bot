from fastapi import FastAPI
from pydantic import BaseModel
from src.chain import med_QA  


app = FastAPI(
    title="RAG_APP",
    description="Retrieval Augmented Generation APP which lets users ask questions and get answers using LLMs",
)

class UserQuestion(BaseModel):
    question: str

class UserAnswer(UserQuestion):
    question: str
    answer: str | None = None

@app.get("/")
def index():
    return {"message": "Hello World"}

@app.post("/questions",response_model=UserAnswer)
def question(user_message: UserQuestion):
    answer = med_QA(user_message.question)
    bot_response= UserAnswer(question=user_message.question,answer=answer)
    return bot_response

