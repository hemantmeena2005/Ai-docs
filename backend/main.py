# main.py

from fastapi import FastAPI, UploadFile, File
import shutil
import os

from rag_pipeline import create_vector_store, ask_question

app = FastAPI()

DATA_PATH = "data"


@app.get("/")
def home():
    return {"message": "AI Study Assistant Backend Running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(DATA_PATH, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create vector DB
    msg = create_vector_store(file_path)

    return {"message": msg}


@app.post("/ask")
async def ask(query: str):
    
    query = query.strip('"')   # remove extra quotes

    answer = ask_question(query)
    return {"question": query, "answer": answer}

from rag_pipeline import generate_summary

@app.get("/summary")
def summary():
    return {"summary": generate_summary()}


from rag_pipeline import generate_quiz

@app.get("/quiz")
def quiz():
    return {"quiz": generate_quiz()}