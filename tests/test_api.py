import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.engine import get_vectorstore
import os

client = TestClient(app)

def test_read_root():
    """Verify the API is alive and reachable."""
    response = client.get("/")
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]

def test_upload_no_file():
    """Verify error handling when no file is provided."""
    response = client.post("/upload")
    assert response.status_code == 422  # Unprocessable Entity (FastAPI default)

def test_invalid_file_type():
    """Verify that non-PDF files are rejected."""
    # Create a fake text file in memory
    files = {"file": ("test.txt", b"not a pdf", "text/plain")}
    response = client.post("/upload", files=files)
    
    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported."    

def test_upload_pdf():
    file_path = os.path.join(os.path.dirname(__file__), "test.pdf")
    with open(file_path, "rb") as f:
        response = client.post("/upload", files={"file": f})
    assert response.status_code == 200
    assert response.json()["status"] == "success"