from fastapi import FastAPI, UploadFile, File
from app.engine import process_document, get_answer, get_vectorstore
import shutil
import os

app = FastAPI()

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Save file locally
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process and index
    vectorstore = process_document(temp_path)
    return {"message": "Document indexed successfully", "file": file.filename}

@app.get("/ask")
async def ask_question(question: str):
    # Load the existing vectorstore from disk
    vectorstore = get_vectorstore()
    
    # Pass the vectorstore and the question to the get_answer function
    response = get_answer(vectorstore, question)
    return {"answer": response}