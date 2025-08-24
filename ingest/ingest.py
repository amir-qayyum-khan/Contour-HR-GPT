import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- CONFIGURATION ---
PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME")
SOURCE_DATA_PATH = "data"

def run_ingestion():
    """
    Loads documents, splits them into chunks, creates embeddings,
    and stores them in Pinecone.
    """
    # Load API keys from secrets
    google_api_key = st.secrets.get("GOOGLE_API_KEY")
    pinecone_api_key = st.secrets.get("PINECONE_API_KEY")

    if not google_api_key or not pinecone_api_key:
        print("API keys not found. Please set them in .streamlit/secrets.toml")
        return

    print("Loading documents...")
    loader = DirectoryLoader(SOURCE_DATA_PATH)
    documents = loader.load()
    if not documents:
        print("No documents found in the data directory.")
        return

    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    print("Initializing embeddings model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_api_key)

    print("Uploading documents and embeddings to Pinecone...")
    PineconeVectorStore.from_documents(docs, embeddings, index_name=PINECONE_INDEX_NAME)
    print("✅ Ingestion complete.")


if __name__ == "__main__":
    run_ingestion()