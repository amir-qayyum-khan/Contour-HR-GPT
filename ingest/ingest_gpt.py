import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

SOURCE_DATA_PATH = "data"

# --- CONFIGURATION ---
load_dotenv(find_dotenv(), override=True)

def run_ingestion():
    """
    Loads documents, splits them into chunks, creates embeddings with OpenAI,
    and stores them in Pinecone.
    """
    # Load API keys from secrets
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not openai_api_key or not pinecone_api_key or not PINECONE_INDEX_NAME:
        print("API keys not found. Please set them in .streamlit/secrets.toml")
        return

    print("Loading documents...")
    loader = DirectoryLoader(SOURCE_DATA_PATH)
    documents = loader.load()
    if not documents:
        print("No documents found in the data directory.")
        return

    print(f"Loaded {len(documents)} documents.")

    # Initialize your connection
    pc = Pinecone(api_key=pinecone_api_key)

    # Get your index object
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Delete all vectors within a specific namespace
    index.delete(delete_all=True)

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    print("Initializing OpenAI embeddings model...")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    print("Uploading documents and embeddings to Pinecone...")
    PineconeVectorStore.from_documents(docs, embeddings, index_name=PINECONE_INDEX_NAME)
    print("✅ Ingestion complete.")


if __name__ == "__main__":
    run_ingestion()