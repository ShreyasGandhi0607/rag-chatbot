import os
from dotenv import load_dotenv

import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain as lc_load_qa_chain
from langchain.prompts import PromptTemplate

from langchain.schema import SystemMessage
from langchain_community.chat_message_histories import FileChatMessageHistory

# ——————————————————————————————————————————————————————————————————————————
# 0. Environment & API Key
# ——————————————————————————————————————————————————————————————————————————
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("⚠️ Please set GOOGLE_API_KEY in your `.env` file and restart.")
    st.stop()

# ——————————————————————————————————————————————————————————————————————————
# 1. PDF ingestion & embedding
# ——————————————————————————————————————————————————————————————————————————
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def build_vectorstore(chunks, api_key):
    embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    store = FAISS.from_texts(chunks, embedding=embed)
    store.save_local("faiss_index")

def load_qa_chain(api_key: str):
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
    return lc_load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def answer_question(query: str, api_key: str):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query)
    qa_chain = load_qa_chain(api_key)
    out = qa_chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    return out["output_text"]

# ——————————————————————————————————————————————————————————————————————————
# 2. History: File-backed chat store
# ——————————————————————————————————————————————————————————————————————————
history_store = FileChatMessageHistory(file_path="chat_history.json")

# If empty, add an initial SystemMessage
if not history_store.messages:
    history_store.add_message(
        SystemMessage(content="You are a helpful assistant.")
    )

# ——————————————————————————————————————————————————————————————————————————
# 3. Streamlit UI
# ——————————————————————————————————————————————————————————————————————————
def main():
    st.set_page_config(page_title="RAGChatbot with History", layout="wide")
    st.title("📚 RAG-Powered PDF Chatbot")

    # Sidebar: Upload & Index PDFs
    st.sidebar.header("1. Upload & Index PDFs")
    pdf_files = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
    if st.sidebar.button("Process & Index PDFs"):
        if not pdf_files:
            st.sidebar.error("Please upload at least one PDF.")
        else:
            with st.spinner("Extracting text and building index…"):
                raw = get_pdf_text(pdf_files)
                chunks = get_text_chunks(raw)
                build_vectorstore(chunks, GOOGLE_API_KEY)
            st.sidebar.success("✅ PDFs indexed!")

    # Main: Ask a Question
    st.header("2. Ask a Question")
    query = st.text_input("Your question about the uploaded PDFs:")
    if st.button("Ask Question"):
        if not os.path.exists("faiss_index"):
            st.error("❗️You need to index PDFs first in the sidebar.")
        elif not query:
            st.error("Please enter a question.")
        else:
            # Record the human turn
            history_store.add_user_message(query)

            # Get the answer
            with st.spinner("Thinking…"):
                answer = answer_question(query, GOOGLE_API_KEY)

            # Record the assistant turn
            history_store.add_ai_message(answer)

            # Display the answer
            st.markdown("**Answer:**")
            st.write(answer)

    # Display full chat history in the sidebar
    st.sidebar.header("Chat History")
    for msg in history_store.messages:
        if msg.__class__.__name__ == "HumanMessage":
            with st.sidebar.chat_message("Human"):
                st.markdown(msg.content)
        elif msg.__class__.__name__ == "AIMessage":
            with st.sidebar.chat_message("AI"):
                st.markdown(msg.content)
        elif msg.__class__.__name__ == "SystemMessage":
            st.sidebar.markdown(f"*System:* {msg.content}")

if __name__ == "__main__":
    main()
