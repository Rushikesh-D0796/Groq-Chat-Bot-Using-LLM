import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# 1. Secret Key Management (Streamlit Cloud + Local fallback)
load_dotenv()

groq_key = None
if "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
else:
    groq_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="PDF QA Assistant", page_icon="📚")
st.title("📚 PDF Question Answering with Groq")
st.write("Upload a PDF file and ask any question from it!")

if not groq_key:
    st.error("GROQ_API_KEY not found! Set it in Streamlit Cloud Secrets or your local .env file.")
    st.stop()

# 2. Cached HuggingFace Embeddings Model
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embedding_model()

# 3. Model Selection
default_models = [
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768"
]

try:
    groq_client = Groq(api_key=groq_key)
    fetched_models = [m.id for m in groq_client.models.list().data if "whisper" not in m.id and "vision" not in m.id]
    model_list = fetched_models if fetched_models else default_models
except Exception:
    model_list = default_models

st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox("Select Groq Model", options=model_list, index=0)
uploaded_pdf = st.sidebar.file_uploader("Upload PDF file", type=["pdf"])

# 4. Chat Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Process Uploaded PDF
if uploaded_pdf is not None:
    temp_filename = "uploaded_document.pdf"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    st.sidebar.success("PDF uploaded successfully!")

    # Step A: Load and Split PDF
    pdf_loader = PyPDFLoader(temp_filename)
    pages = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(pages)

    # Step B: Build Vector Store
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # Step C: Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Step D: Chat Input
    user_question = st.chat_input("Ask something from the document...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Step E: Retrieve Context
        matching_docs = retriever.invoke(user_question)
        context_text = "\n\n".join([doc.page_content for doc in matching_docs]) if matching_docs else "No context found."

        # Step F: Universal Prompt (Avoids system-role rejection on Groq models)
        prompt_template = ChatPromptTemplate.from_messages([
            (
                "human",
                "You are a strict QA assistant. Answer the user question based ONLY on the provided context below.\n"
                "If the answer is not in the context, say 'I cannot find that in the document.'\n\n"
                "--- CONTEXT ---\n{context}\n\n"
                "--- QUESTION ---\n{question}"
            )
        ])

        final_prompt = prompt_template.format_messages(
            context=context_text,
            question=user_question
        )

        # Step G: Call Groq API
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                groq_model = ChatGroq(
                    groq_api_key=groq_key,
                    model=selected_model,
                    temperature=0.1
                )
                response = groq_model.invoke(final_prompt)
                st.write(response.content)

        st.session_state.messages.append({"role": "assistant", "content": response.content})

else:
    st.info("Please upload a PDF file from the left sidebar to start.")