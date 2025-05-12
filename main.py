import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores import FAISS,Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence  

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# 1. Load environment config
def get_config():
    chroma_dir = os.getenv("CHROMA_DIR", "./chroma_db")
    # GOOGLE_APPLICATION_CREDENTIALS must be set in env for Gemini
    return chroma_dir

# 2. Ingest and index documents
def load_and_index(path, embeddings, persist_dir=None):
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    if persist_dir:
        return Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    return FAISS.from_documents(chunks, embeddings)

# 3. Initialize vector stores for static reports
@st.cache_resource
def init_stores():
    chroma_dir = get_config()
    # Use Google Gemini embeddings for vectorization
    embeddings = GooglePalmEmbeddings(google_api_key=api_key)
    apple_store = load_and_index(
        "documents/10Q-Q1-2025-as-filed.pdf", embeddings,
        os.path.join(chroma_dir, "apple")
    )
    dow_store = load_and_index(
        "documents/Dow_2024_Annual_Report_Web.pdf", embeddings,
        os.path.join(chroma_dir, "dowjones")
    )
    return embeddings, apple_store, dow_store

# 4. Build a RAG chain using RunnableSequence and Gemini Flash

def build_rag_chain(store):
    # Retriever: fetch top-k docs
    retriever = store.as_retriever(search_kwargs={"k": 4})
    # Prompt template merges context and question
    rag_prompt = PromptTemplate(
        template=(
            "Use the following financial report excerpts to answer the question.\n"
            "Context:\n{context}\n"
            "Question: {question}\n"
            "Answer in a concise, informative style."
        ),
        input_variables=["context", "question"]
    )
    # Parser to extract string
    parser = StrOutputParser()
    # Gemini Flash model
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)
    # Runnable chain: prompt -> model -> parser
    return rag_prompt | model | parser, retriever

# 5. Streamlit app UI

def main():
    st.set_page_config(page_title="Finance RAG Chatbot", page_icon="💬")
    st.title("📈 RAG Chatbot: Apple & Dow-Jones (Gemini Flash)")

    # Sidebar: data source selection
    source = st.sidebar.radio(
        "Data Source", ["Apple Q1", "Dow-Jones", "Upload PDF"]
    )

    embeddings, apple_store, dow_store = init_stores()

    if source == "Apple Q1":
        store = apple_store
    elif source == "Dow-Jones":
        store = dow_store
    else:
        uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
        if not uploaded:
            st.info("Please upload a document to proceed.")
            return
        temp_path = os.path.join("/tmp", uploaded.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        store = load_and_index(temp_path, embeddings)

    # Build RAG chain
    rag_chain, retriever = build_rag_chain(store)

    # Session history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Render chat history
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if query := st.chat_input("Ask about financial reports..."):
        st.session_state.history.append({"role": "user", "content": query})
        docs = retriever.get_relevant_documents(query)
        context = "\n---\n".join([d.page_content for d in docs])
        answer = rag_chain.invoke({"context": context, "question": query})
        st.session_state.history.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

if __name__ == "__main__":
    main()