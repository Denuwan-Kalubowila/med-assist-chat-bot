import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from typing import List, Any, Optional
from pydantic import BaseModel

# Import your FastAPI app and models
from src.main import app, UserQuestion, UserAnswer

# Create test client
client = TestClient(app)

class TestMedicalAssistantAPI:
    @pytest.fixture
    def mock_med_qa(self):
        """Fixture for mocking the medical QA function"""
        with patch('src.main.med_QA') as mock:
            # Mock response with sample answer and chat history
            mock.return_value = (
                "This is a test medical response",
                [
                    {"role": "user", "content": "What are the symptoms of a cold?"},
                    {"role": "assistant", "content": "Common cold symptoms include..."}
                ]
            )
            yield mock

    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello World"}

    def test_question_endpoint_success(self, mock_med_qa):
        """Test successful question submission"""
        test_question = {"question": "What are the symptoms of a cold?"}
        
        response = client.post("/questions", json=test_question)
        
        assert response.status_code == 200
        assert response.json()["question"] == test_question["question"]
        assert "answer" in response.json()
        assert "chat_history" in response.json()
        assert isinstance(response.json()["chat_history"], list)

    def test_question_endpoint_empty_question(self):
        """Test handling of empty question"""
        test_question = {"question": ""}
        
        response = client.post("/questions", json=test_question)
        
        assert response.status_code == 200  # Validation error

    def test_question_endpoint_long_question(self, mock_med_qa):
        """Test handling of very long question"""
        test_question = {"question": "a" * 1000}
        
        response = client.post("/questions", json=test_question)
        
        assert response.status_code == 200
        assert "answer" in response.json()

    def test_question_endpoint_special_characters(self, mock_med_qa):
        """Test handling of special characters in question"""
        test_question = {"question": "What about <!@#$%^&*()> symbols?"}
        
        response = client.post("/questions", json=test_question)
        
        assert response.status_code == 200
        assert "answer" in response.json()

    def test_question_endpoint_invalid_json(self):
        """Test handling of invalid JSON"""
        response = client.post(
            "/questions",
            json={"invalid_field": "test"}
        )
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_med_qa):
        """Test handling of concurrent requests"""
        import asyncio
        import httpx
        
        async def make_request():
            async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
                response = await ac.post(
                    "/questions",
                    json={"question": "Test question"}
                )
                return response

        # Make 5 concurrent requests
        responses = await asyncio.gather(
            *[make_request() for _ in range(5)]
        )
        
        for response in responses:
            assert response.status_code == 200
            assert "answer" in response.json()

    def test_response_model_validation(self, mock_med_qa):
        """Test if response matches UserAnswer model"""
        test_question = {"question": "Test question"}
        
        response = client.post("/questions", json=test_question)
        
        # Validate response against UserAnswer model
        user_answer = UserAnswer(**response.json())
        assert isinstance(user_answer.question, str)
        assert isinstance(user_answer.answer, str)
        assert isinstance(user_answer.chat_history, list)


if __name__ == "__main__":
    pytest.main([__file__])