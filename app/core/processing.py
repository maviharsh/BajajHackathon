# core/processing.py

import os
import tempfile
import requests # Add this import
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredEmailLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- Mapping from file extensions to loaders remains the same ---
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".txt": TextLoader,
    ".eml": UnstructuredEmailLoader,
}

def load_document(filepath: str):
    """Loads a single document from a file path using the correct loader."""
    ext = "." + filepath.rsplit(".", 1)[-1].lower()
    if ext in LOADER_MAPPING:
        loader = LOADER_MAPPING[ext](filepath)
        return loader.load()
    raise ValueError(f"Unsupported file type: {ext}")

def download_and_process_document(url: str):
    """
    Downloads a document from a URL, processes it, and returns text chunks.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Get file extension from URL
        file_ext = os.path.splitext(url)[1].lower()

        # Create a temporary file to store the downloaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(response.content)
            tmp_filepath = tmp_file.name
        
        # Load the document from the temporary file
        documents = load_document(tmp_filepath)
        
        # Clean up the temporary file
        os.remove(tmp_filepath)
        
        if not documents:
            return []

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        split_texts = text_splitter.split_documents(documents)
        
        return split_texts

    except requests.exceptions.RequestException as e:
        print(f"Error downloading document from {url}: {e}")
        return []
    except Exception as e:
        print(f"Error processing document from {url}: {e}")
        return []