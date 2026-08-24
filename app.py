import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# LangChain and Groq imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load API key from .env file
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

#  Page Title
st.title("📚 PDF Question Answering with Groq")
st.write("Upload a PDF file and ask any question from it!")

# Check if key exists
if not groq_key:
    st.error("GROQ_API_KEY not found! Please check your .env file.")
    st.stop()

# Fetch active models dynamically for your key
try:
    groq_client = Groq(api_key=groq_key)
    model_list = [m.id for m in groq_client.models.list().data if "whisper" not in m.id]
except Exception as e:
    # Fallback list if fetching fails
    model_list = ["llama3-8b-8192", "llama3-70b-8192", "gemma2-9b-it", "mixtral-8x7b-32768"]

# Sidebar Controls
st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox("Select Groq Model", options=model_list)
uploaded_pdf = st.sidebar.file_uploader("Upload PDF file", type=["pdf"])

#  Save chat history in session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Process the PDF when uploaded
if uploaded_pdf is not None:
    temp_filename = "uploaded_document.pdf"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    st.sidebar.success("PDF uploaded successfully!")

    # Load text from PDF
    pdf_loader = PyPDFLoader(temp_filename)
    pages = pdf_loader.load()

    #  Split document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(pages)

    #  Load Embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create FAISS vector database
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # Display past chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    #  User Question Input
    user_question = st.chat_input("Ask something from the document...")

    if user_question:
        # Show user question in chat
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        #  Retrieve relevant text from vector store
        matching_docs = retriever.invoke(user_question)
        context_text = ""
        for doc in matching_docs:
            context_text = context_text + doc.page_content + "\n\n"

        #  Build strict prompt
        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant. Answer the question using ONLY the provided document context below. "
                "If the answer is not in the context, say 'I cannot find that in the document.'\n\n"
                "Context:\n{context}"
            ),
            ("human", "{question}")
        ])

        final_prompt = prompt_template.format_messages(
            context=context_text,
            question=user_question
        )

        #  Get answer from selected Groq Model
        groq_model = ChatGroq(
            groq_api_key=groq_key,
            model=selected_model,
            temperature=0.2
        )
        response = groq_model.invoke(final_prompt)

        # Show assistant answer in chat
        with st.chat_message("assistant"):
            st.write(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})

else:
    st.info("Please upload a PDF file from the left sidebar to start.")