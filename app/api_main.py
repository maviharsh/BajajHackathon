# api_main.py

import os
import sys
from fastapi import FastAPI, Security, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- This is the fix for ChromaDB on deployed environments ---
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# -----------------------------------------------------------

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Import your existing, modified logic
from .core.processing import download_and_process_document
from .core.rag_handler import get_structured_rag_response

# --- 1. Initialize Application and Load Environment Variables ---
app = FastAPI(
    title="Document Intelligence API",
    description="API for processing documents and answering questions based on their content.",
    version="1.0.0"
)
load_dotenv()

# --- 2. API Key Authentication Setup ---
API_KEY = os.getenv("API_KEY") # This is the key your API will expect
if not API_KEY:
    raise ValueError("API_KEY not found in .env file. This is for securing your API endpoint.")
    
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """Validates the API key from the Authorization header."""
    if not api_key_header or api_key_header != f"Bearer {API_KEY}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )
    return api_key_header

# --- 3. Define Request and Response Models (as per hackathon spec) ---
class HackathonRequest(BaseModel):
    documents: list[str] = Field(..., description="A list of public URLs to the documents.")
    questions: list[str] = Field(..., description="A list of questions to ask about the documents.")

class HackathonResponse(BaseModel):
    answers: list[str] = Field(..., description="A list of answers, corresponding to each question.")

# --- 4. Define the Main API Endpoint ---
@app.post("/hackrx/run", response_model=HackathonResponse)
async def run_document_query(
    request: HackathonRequest, 
    api_key: str = Security(get_api_key)
):
    """
    Processes documents from URLs, answers questions, and returns the results.
    """
    all_chunks = []
    
    # --- A. Process all documents and create a unified context ---
    for url in request.documents:
        print(f"Processing document: {url}")
        chunks = download_and_process_document(url)
        if chunks:
            all_chunks.extend(chunks)
        else:
            # You might want to raise an error if a document fails to load
            print(f"Warning: Failed to process document from {url}")

    if not all_chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents could be processed from the provided URLs."
        )

    # --- B. Create a single in-memory vector store for this request ---
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = Chroma.from_documents(documents=all_chunks, embedding=embeddings)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create vector store: {e}"
        )

    # --- C. Iterate through questions and generate answers ---
    final_answers = []
    for question in request.questions:
        print(f"Answering question: {question}")
        # Use your existing rag_handler to get the detailed, structured response
        response_dict = get_structured_rag_response(question, vector_store)
        
        if "error" in response_dict:
            # Format the error nicely for the final response list
            answer_str = f"Error processing question '{question}': {response_dict['error']}"
        else:
            # Format the structured JSON into the simple string required by the hackathon
            decision = response_dict.get('decision', 'N/A')
            amount = response_dict.get('amount', 0.0)
            justification = response_dict.get('justification', 'No justification available.')
            # Combine into a single string. You can format this however you like.
            answer_str = f"Decision: {decision}, Amount: {amount:.2f}, Justification: {justification}"

        final_answers.append(answer_str)

    # --- D. Return the final response in the specified format ---
    return HackathonResponse(answers=final_answers)

# To run this API locally:
# uvicorn api_main:app --reload
