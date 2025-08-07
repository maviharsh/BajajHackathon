import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredEmailLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import shutil

# --- 1. Load Environment Variables ---
print("Loading environment variables...")
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# --- 2. Define Constants ---
RAW_DOCS_PATH = "data/raw"
PERSIST_DIRECTORY = "data/vector_store/chromadb"

# --- 3. Mapping from file extensions to loaders ---
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".txt": TextLoader,
    ".eml": UnstructuredEmailLoader,
}

# --- 4. Document Loading Function (Upgraded) ---
def load_documents_from_directory(directory_path):
    """
    Loads all supported documents from a specified directory using the correct loader.
    """
    all_documents = []
    print(f"Loading documents from: {directory_path}")
    if not os.path.exists(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return []

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        # Get the file extension and find the appropriate loader
        file_ext = os.path.splitext(filename)[1].lower()
        loader_class = LOADER_MAPPING.get(file_ext)

        if loader_class:
            try:
                print(f"Loading {filename} with {loader_class.__name__}...")
                # Initialize the loader with the file path
                loader = loader_class(file_path)
                # Load the documents
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"Skipping {filename}: Unsupported file type.")

    return all_documents

# --- 5. Main Ingestion Logic ---
def main():
    """
    Main function to run the data ingestion pipeline.
    """
    # Clean out the old database first for a fresh start.
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"Removing old vector store at {PERSIST_DIRECTORY}")
        shutil.rmtree(PERSIST_DIRECTORY)

    # Load all documents from the source directory.
    documents = load_documents_from_directory(RAW_DOCS_PATH)
    if not documents:
        print("No documents were loaded. Please check the 'data/raw' directory. Exiting.")
        return

    # Split documents into chunks.
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    split_texts = text_splitter.split_documents(documents)
    print(f"Created {len(split_texts)} text chunks.")

    # Initialize embedding model.
    print("Initializing OpenAI embedding model...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create and persist the vector store.
    print(f"Creating and persisting vector store at: {PERSIST_DIRECTORY}")
    db = Chroma.from_documents(
        documents=split_texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    print("\n--- Ingestion Complete ---")
    print(f"Vector store has been successfully created at '{PERSIST_DIRECTORY}'.")

if __name__ == "__main__":
    main()
