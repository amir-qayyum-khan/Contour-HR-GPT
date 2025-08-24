import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores.pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore as Pinecone

# --- CONFIGURATION ---
PINECONE_INDEX_NAME = "hr-document-chatbot-prod"

# --- API and DB SETUP ---
def configure_apis():
    """Load API keys from Streamlit secrets and configure clients."""
    google_api_key = st.secrets.get("GOOGLE_API_KEY")
    pinecone_api_key = st.secrets.get("PINECONE_API_KEY")

    if not google_api_key or not pinecone_api_key:
        st.error("API keys not found. Please add them to your Streamlit secrets.")
        st.stop()
    
    genai.configure(api_key=google_api_key)
    return google_api_key, pinecone_api_key

def get_vectorstore():
    """Initializes and returns the Pinecone vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    # This is deprecated, but we are using it to avoid installing langchain-pinecone
    vector_store = Pinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    return vector_store

def get_gemini_response(question, context):
    """Generates a response from Gemini using the provided context."""
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    prompt = f"""
    You are a helpful and polite HR assistant for Contour Software.
    - If the user provides a greeting or engages in small talk, respond naturally and professionally.
    - For questions about HR policy, answer based *only* on the provided context.
    - If the context does not contain the answer to a policy question, state that the information isn't available in the documents and suggest contacting the HR department for more details.

    CONTEXT:
    {context}

    QUESTION:
    {question}
    """
    response = model.generate_content(prompt)
    return response.text

# --- STREAMLIT APP ---
def main():
    st.set_page_config(page_title="🔍 HR GPT", layout="wide")
    st.title("🔍 Contour HR GPT")
    st.write("Welcome to the Contour HR Policy Assistant. Ask any question about our company policies and procedures.")

    # Configure APIs
    configure_apis()

    # Initialize vector store
    try:
        vector_store = get_vectorstore()
    except Exception as e:
        st.error(f"Failed to connect to Pinecone. Please check your index name and API keys. Error: {e}")
        st.stop()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is your question?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant", avatar="assets/logo.png"):
            with st.spinner("Thinking..."):
                # Find relevant documents
                docs = vector_store.similarity_search(prompt, k=3)
                context = "\n".join([doc.page_content for doc in docs])
                
                # Generate response
                response = get_gemini_response(prompt, context)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()