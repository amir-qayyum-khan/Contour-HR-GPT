# Contour HR GPT

Contour HR GPT is an AI-powered assistant for querying company HR policies and procedures. It uses Google Gemini for natural language understanding and Pinecone for semantic search over your HR documents.

## Features

- Ask questions about HR policies and procedures.
- Retrieves relevant information from uploaded PDF documents.
- Uses Google Gemini for accurate, context-based answers.
- Streamlit-based interactive chat interface.

## Setup

1. **Clone the repository**  

2. **Install dependencies**  

3. **Add your API keys**  

- Edit `.streamlit/secrets.toml` and add your `GOOGLE_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_INDEX_NAME`.

4. **Add HR documents**  

- Place your PDF policy documents in the `data/` folder.

5. **Ingest documents**  

6. **Run the app**  


## File Structure

- `app.py` — Streamlit app for chat interface.
- `ingest.py` — Script to process and upload documents to Pinecone.
- `data/` — Folder for HR policy PDFs.
- `.streamlit/secrets.toml` — API keys and configuration.
- `requirements.txt` — Python dependencies.

## Usage

- Open the app in your browser.
- Type your HR-related question in the chat.
- The assistant will search your documents and provide an answer.

## License

Proprietary. For internal use at Contour Software.
