import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredEmailLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- Mapping from file extensions to loaders ---
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".txt": TextLoader,
    ".eml": UnstructuredEmailLoader,
}

def load_document(temp_filepath: str):
    """
    Loads a single document from a temporary file path using the correct loader.
    """
    # Get the file extension to determine the loader
    file_ext = os.path.splitext(temp_filepath)[1].lower()
    loader_class = LOADER_MAPPING.get(file_ext)

    if loader_class:
        try:
            loader = loader_class(temp_filepath)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"Error loading file {temp_filepath}: {e}")
            return None
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

def process_document_for_rag(temp_filepath: str):
    """
    Processes a single uploaded document and returns an in-memory vector store.
    """
    documents = load_document(temp_filepath)
    if not documents:
        return None

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    split_texts = text_splitter.split_documents(documents)

    # Initialize embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create an in-memory Chroma vector store
    vector_store = Chroma.from_documents(documents=split_texts, embedding=embeddings)
    
    return vector_store
