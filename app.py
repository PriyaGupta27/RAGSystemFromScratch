import os
from uuid import uuid4

import chromadb
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()
requests_session = requests.Session()
client = Groq(api_key=os.getenv("AGENT_KEY"))

# Load document -> Chunking -> Embedding -> Vector DB -> Query -> Retrieve

#----------------------Data Ingestion-------------------------
def extract_text_from_url(url: str) -> str:
    try:
        response = requests_session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        return " ".join(soup.get_text(separator=" ").split())
    except Exception as e:
        return f"Error fetching URL: {e}"
    
#----------------------------Chunking-------------------------------
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50):
    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]

#------------------------Embedding---------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

#--------------------------VectorDB---------------------------------
@st.cache_resource
def init_chroma():
    chroma_client = chromadb.Client()
    return chroma_client.get_or_create_collection(name="rag_collection")

collection = init_chroma()

#--------------------------Retrieval--------------------------------
def retrieve_chunks(query, k=3):
    query_embedding = embedding_model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    return(
        results["documents"][0],
        results["metadatas"][0]
    )

#------------------------PROMPT-------------------------------------
def build_prompt(context_chunks, query):
    context = "\n\n".join(context_chunks)

    return f"""
    You are an AI assistant. Answer strictly based on the provided context.
    If the answer is not in the context, say "I don't know."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

#---------------------------GROQ------------------------------------
def query_groq(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role":"system", "content": "You are a helpful AI assistant."},
            {"role":"user", "content":prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

#----------------------------UI-------------------------------------
    
st.header("RAG System")

if "documents" not in st.session_state:
    st.session_state.documents = []

urls_input = st.sidebar.text_area("Enter URLS in separate lines")

if st.sidebar.button("Load URLs"):
    urls = [u.strip() for u in urls_input.splitlines() if u.strip()]

    for url in urls:
        text = extract_text_from_url(url)
        if text and not text.startswith("Error"):
            st.session_state.documents.append(
                {"source": url, "type": "url", "content": text}
            )

    if not st.session_state.documents:
        st.warning("No documents loaded.")
    else:
        all_chunks = []
        metadatas = []
        ids = []

        for doc in st.session_state.documents:
            for chunk in chunk_text(doc["content"]):
                all_chunks.append(chunk)
                metadatas.append({"source": doc["source"], "type": doc["type"]})
                ids.append(str(uuid4()))

        embeddings = embedding_model.encode(all_chunks).tolist()
        collection.add(
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

if st.session_state.documents:
    st.sidebar.subheader("Loaded Documents")
    st.sidebar.write(st.session_state.documents[0]["content"][:580])


messages = st.container(height=200)
if query := st.chat_input("Enter a query"):
    messages.chat_message('user').write(query)
    chunks, metadata = retrieve_chunks(query)
    prompt = build_prompt(chunks, query)
    answer = query_groq(prompt)
    messages.chat_message('assistant').write(answer)
