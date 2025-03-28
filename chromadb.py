import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from chromadb.config import Settings
client_settings = Settings(anonymized_telemetry=False)

# --- Load .env variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Config ---
embedding_model = "text-embedding-3-small"
doc_dir = "documents"
persist_base_dir = "./chroma_db"
chunk_size = 800
chunk_overlap = 200
top_k = 5

os.makedirs(persist_base_dir, exist_ok=True)
st.set_page_config(page_title="Grant QA System", layout="wide")
st.title("📑 Grant Document Q&A Assistant")

# --- Utility: load & chunk all files ---
def load_documents():
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    file_chunks = {}

    for filename in os.listdir(doc_dir):
        path = os.path.join(doc_dir, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue

        docs = loader.load()
        chunks = splitter.split_documents(docs)
        for i, doc in enumerate(chunks):
            doc.metadata["filename"] = filename
            doc.metadata["chunk_id"] = i
        file_chunks[filename] = chunks

    return file_chunks

file_chunks = load_documents()
doc_names = list(file_chunks.keys())
selected_doc = st.selectbox("Select a document to query:", doc_names)
query = st.text_input("Ask a question about the selected document:")

# --- Helper: persistent Chroma store per document ---
def get_or_create_vectorstore(doc_name, docs_split):
    embedding = OpenAIEmbeddings(model=embedding_model, openai_api_key=OPENAI_API_KEY)
    collection_name = doc_name.replace(" ", "_").replace(".pdf", "").replace(".docx", "")
    persist_dir = os.path.join(persist_base_dir, collection_name)

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        return Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=embedding,
            client_settings=client_settings
        )
    else:
        vectorstore = Chroma.from_documents(
            docs_split,
            embedding,
            collection_name=collection_name,
            persist_directory=persist_dir,
            client_settings=client_settings
        )
        vectorstore.persist()
        return vectorstore

# --- Main QA Block ---
if query:
    with st.spinner("🔍 Thinking..."):
        docs_split = file_chunks[selected_doc]
        vectorstore = get_or_create_vectorstore(selected_doc, docs_split)
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

        # Prompt setup
        chat_prompt = ChatPromptTemplate.from_template("""
You are an intelligent assistant supporting a nonprofit organization in drafting effective grant proposals.

Your job is to answer questions **based only on the context provided**, which comes from specific grant-related documents (e.g., fellowship proposals, grant calls, previous applications).

Respond clearly and concisely. If the answer is not present in the provided context, respond with:

"I don’t know based on the document."

Avoid making assumptions. Do not use any external knowledge or generate answers from general understanding — rely strictly on the context.

Context:
{context}

Question: {question}
""")
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

        results = retriever.get_relevant_documents(query)
        context = "\n\n---\n\n".join([doc.page_content for doc in results])
        prompt = chat_prompt.format(context=context, question=query)
        response = llm.predict(prompt)

        # Output
        st.subheader("🧠 Answer")
        st.markdown(response)

        with st.expander("📄 Source Chunks"):
            for doc in results:
                st.markdown(f"**[Chunk {doc.metadata.get('chunk_id')}]**")
                st.code(doc.page_content.strip()[:1000])
                st.markdown(f"*Source: {doc.metadata.get('filename')}*")