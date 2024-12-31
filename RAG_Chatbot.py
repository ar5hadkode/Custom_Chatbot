import streamlit as st
import os
import pickle
import pdfplumber
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from groq import Groq

def extract_text_chunks(pdf_path, chunk_size=1000, chunk_overlap=100):
    with pdfplumber.open(pdf_path) as pdf:
        text = "".join(page.extract_text() or "" for page in pdf.pages)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def create_and_save_vector_database(chunks, db_path):
    embedding_model = 'all-MiniLM-L6-v2'
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    with open(db_path, "wb") as f:
        pickle.dump(vectorstore, f)
    return vectorstore

def load_vector_database(db_path): 
    with open(db_path, "rb") as f:
        return pickle.load(f)

def retrieve_relevant_chunks(query, vectorstore, top_k=5):
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embedding_model.encode(query)
    docs = vectorstore.similarity_search_by_vector(query_embedding, k=top_k)
    return [doc.page_content for doc in docs]

def query_groq_basic(user_query):
    api_key = "YOUR_GROQ_API_KEY"
    client = Groq(api_key=api_key)
    prompt = (
        f"You are an intelligent assistant. Answer the user's query concisely and accurately: '{user_query}'."
        "\nIf you don't have knowledge about something, just write 'It's out of my knowledge. "
    )
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content

def query_groq_rag(relevant_chunks, user_query):
    api_key = "YOUR_GROQ_API_KEY"
    client = Groq(api_key=api_key)
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an intelligent assistant. Below are the relevant excerpts from my knowledge:\n\n" 
        + context +
        "\n\nBased on this, answer the user's query: '" + user_query + "'. Provide a concise and accurate response."
        "\nIf you don't have knowledge about something, just write 'It's out of my knowledge. Just answer directly, don't write like 'Based on the provided excerpts'"
    )
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content

                                # Streamlit App ............................................
                                    
st.set_page_config(page_title="AgileKode Personal ChatBot", layout="wide")
st.title("AgileKode Personal ChatBot")

if "history_basic" not in st.session_state:
    st.session_state.history_basic = []
if "history_rag" not in st.session_state:
    st.session_state.history_rag = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "mode" not in st.session_state:              
    st.session_state.mode = "Ask Me Anything"

model_options = ["Ask Me Anything", "Ask about AgileKode"]
st.sidebar.write("### Select your LLM")
st.session_state.mode = st.sidebar.radio("Options", model_options, index=0 if st.session_state.mode == "Ask Me Anything" else 1)

st.markdown(f"### Current Mode: **{st.session_state.mode}**")
st.write("---")


if st.session_state.mode == "Ask about AgileKode":
    PDF_PATH = "agilekode-portfolio.pdf"
    VECTOR_DB_PATH = "vectorstore.pkl"

    if not os.path.exists(VECTOR_DB_PATH):
        st.write("Creating vector database from the PDF. This will take a moment...")
        chunks = extract_text_chunks(PDF_PATH)
        vectorstore = create_and_save_vector_database(chunks, VECTOR_DB_PATH)
        st.success("Vector database created and saved.")
    else:
        vectorstore = load_vector_database(VECTOR_DB_PATH)

    user_query = st.text_input("Ask me a question:", key="user_input", placeholder="Type your question here...")
    if user_query:
        relevant_chunks = retrieve_relevant_chunks(user_query, vectorstore)
        response = query_groq_rag(relevant_chunks, user_query)
        st.session_state.history_rag.append({"query": user_query, "response": response})
        st.session_state.last_query = ""  # Clear input box

    for entry in reversed(st.session_state.history_rag):
        st.markdown(f"**You:** {entry['query']}")
        st.markdown(f"**Assistant:** {entry['response']}")
        st.write("---")
else:
    user_query = st.text_input("Ask me a question:", key="basic_input", placeholder="Type your question here...")
    if user_query:
        response = query_groq_basic(user_query)
        st.session_state.history_basic.append({"query": user_query, "response": response})
        st.session_state.last_query = ""  # Clear input box

    for entry in reversed(st.session_state.history_basic):
        st.markdown(f"**You:** {entry['query']}")
        st.markdown(f"**Assistant:** {entry['response']}")
        st.write("---")

# ----------------------------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------------------------#
