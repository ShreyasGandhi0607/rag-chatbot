import os
from dotenv import load_dotenv

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain as lc_load_qa_chain
from langchain.prompts import PromptTemplate

# ——————————————————————————————————————————————————————————————————————————
# 1. Load your .env file and grab the key at program start (before Streamlit inputs)
# ——————————————————————————————————————————————————————————————————————————
load_dotenv()  
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("⚠️ Please set GOOGLE_API_KEY in your `.env` file and restart.")
    st.stop()

# ——————————————————————————————————————————————————————————————————————————
# 2. Helper functions
# ——————————————————————————————————————————————————————————————————————————

def get_pdf_text(pdf_files):
    """Extract all text from an uploaded list of PDFs."""
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
    return text

def get_text_chunks(text: str):
    """Split the raw text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def build_vectorstore(chunks, api_key):
    """Embed chunks and build a FAISS index, saving it locally."""
    embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    store = FAISS.from_texts(chunks, embedding=embed)
    store.save_local("faiss_index")

def load_qa_chain(api_key: str):
    """
    Returns a loaded QA chain using Gemini-Pro,
    safely aliased so we can pass chain_type!
    """
    prompt_template = """
Answer the question as detailed as possible from the provided context.
If the answer is not in the provided context, just say "answer is not available in the context".

Context:
{context}

Question:
{question}

Answer:
"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=api_key
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    # call the *aliased* LangChain function that accepts chain_type
    return lc_load_qa_chain(
        llm,
        chain_type="stuff",
        prompt=prompt
    )

def answer_question(query: str, api_key: str):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    docs = db.similarity_search(query)
    qa_chain = load_qa_chain(api_key)  # now your helper calls lc_load_qa_chain correctly
    out = qa_chain(
        {"input_documents": docs, "question": query},
        return_only_outputs=True
    )
    return out["output_text"]


# ——————————————————————————————————————————————————————————————————————————
# 3. Streamlit UI
# ——————————————————————————————————————————————————————————————————————————

def main():
    st.set_page_config(page_title="RAGChatbot", layout="wide")
    st.title("📚 RAG-Powered PDF Chatbot")

    # — Sidebar: PDF upload and indexing
    st.sidebar.header("1. Upload & Index PDFs")
    pdf_files = st.sidebar.file_uploader(
        "Upload PDF files", accept_multiple_files=True, type=["pdf"]
    )
    if st.sidebar.button("Process & Index PDFs"):
        if not pdf_files:
            st.sidebar.error("Please upload at least one PDF.")
        else:
             with st.spinner("Extracting text and building index…"):
                raw = get_pdf_text(pdf_files)
                chunks = get_text_chunks(raw)
                build_vectorstore(chunks, GOOGLE_API_KEY)
        st.sidebar.success("✅ PDFs indexed!")

    # — Main: Ask a question
    st.header("2. Ask a Question")
    query = st.text_input("Your question about the uploaded PDFs:")
    if st.button("Ask Question"):
        if not os.path.exists("faiss_index"):
            st.error("❗️You need to index PDFs first in the sidebar.")
        elif not query:
            st.error("Please enter a question.")
        else:
            with st.spinner("Thinking…"):
                answer = answer_question(query, GOOGLE_API_KEY)
            st.markdown("**Answer:**")
            st.write(answer)

if __name__ == "__main__":
    main()
