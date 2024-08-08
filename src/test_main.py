from fastapi.testclient import TestClient
from src.main import app

client =TestClient(app)

def test_question_main():
    payload = {
        "question": "What is the treatment for a yellow fever?"
    }
    res = client.post("/questions", json=payload)
    assert res.status_code == 200
    
