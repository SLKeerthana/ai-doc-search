from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from typing import List
from app.engine import process_document, get_answer, get_vectorstore
import shutil
import os

app = FastAPI(
    title="AI Document Intelligence API",
    description="A production-grade RAG API for PDF summarization and semantic search.",
    version="1.0.0"
)

# # --- Schemas (For clean Documentation) ---
# class QueryResponse(BaseModel):
#     question: str
#     answer: str
#     sources: List[int]

# # --- Endpoints ---

@app.post("/upload",
tags=["Document Management"],
summary="Upload and Index a PDF"
)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Uploads a PDF file, extracts text, generates embeddings using Ollama, 
    and persists them to the ChromaDB vector store.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # Save file locally
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Process and index
        vectorstore = process_document(temp_path)
        return {"status": "success", "message": "Document indexed successfully", "file": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/ask", tags=["AI Operations"], summary="Query the knowledge Base")
async def ask_question(question: str = Query(..., examples="What is the main conclusion of the document?")):
    """
    Performs a semantic search across all indexed documents and returns 
    an AI-generated answer based on relevant context.
    """
    # Load the existing vectorstore from disk
    vectorstore = get_vectorstore()
    if not vectorstore:
        raise HTTPException(status_code=404, detail="No documents indexed yet.")
    
    # Pass the vectorstore and the question to the get_answer function
    response = get_answer(vectorstore, question)
    return {"answer": response}