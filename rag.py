import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --- Load .env ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Config ---
embedding_model = "text-embedding-3-small"
doc_dir = "documents"
persist_base_dir = "./faiss_db"
chunk_size = 800
chunk_overlap = 200
top_k_initial = 15
top_k_final = 5

st.set_page_config(page_title="Grant QA System", layout="wide")
st.title("📑 Grant Document Q&A Assistant")

# --- Document Loading ---
@st.cache_resource
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

# --- Vectorstore Creation / Loading ---
def get_or_create_faiss(doc_name, docs_split):
    embedding = OpenAIEmbeddings(model=embedding_model, openai_api_key=OPENAI_API_KEY)
    faiss_path = os.path.join(persist_base_dir, doc_name.replace(" ", "_"))

    if os.path.exists(faiss_path):
        return FAISS.load_local(faiss_path, embeddings=embedding, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(docs_split, embedding)
        vectorstore.save_local(faiss_path)
        return vectorstore

# --- Reranking Function ---
def rerank_chunks_with_llm(question, docs: list[Document], top_n=5):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
    scored_docs = []

    for doc in docs:
        prompt = f"""
You are a helpful assistant. Score how relevant the following chunk is for answering the user's question.

Respond with a number from 0 to 10.

Chunk:
\"\"\"
{doc.page_content}
\"\"\"

Question: {question}

Score:"""
        try:
            score = float(llm.predict(prompt).strip())
        except:
            score = 0.0
        scored_docs.append((score, doc))

    scored_docs.sort(reverse=True, key=lambda x: x[0])
    return [doc for score, doc in scored_docs[:top_n]]

# --- Load & Setup ---
file_chunks = load_documents()
doc_names = list(file_chunks.keys())
selected_doc = st.selectbox("Select a document to query:", doc_names)
query = st.text_input("Ask a question about the selected document:")

# --- Main QA Pipeline ---
if query:
    with st.spinner("🔍 Thinking..."):
        docs_split = file_chunks[selected_doc]
        vectorstore = get_or_create_faiss(selected_doc, docs_split)
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k_initial})
        retrieved_docs = retriever.get_relevant_documents(query)

        reranked_docs = rerank_chunks_with_llm(query, retrieved_docs, top_n=top_k_final)

        # Format citations
        citations = []
        context_blocks = []
        for doc in reranked_docs:
            filename = doc.metadata.get("filename", "unknown")
            chunk_id = doc.metadata.get("chunk_id", "n/a")
            context_blocks.append(doc.page_content)
            citations.append(f"[Chunk {chunk_id}] — *{filename}*")

        context = "\n\n---\n\n".join(context_blocks)
        citation_text = "\n".join(citations)

        prompt_template = ChatPromptTemplate.from_template("""
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
        prompt = prompt_template.format(context=context, question=query)
        response = llm.predict(prompt)

        # --- Output ---
        st.subheader("🧠 Answer")
        st.markdown(response)

        st.markdown("#### 📄 Citations")
        for doc in reranked_docs:
            chunk_id = doc.metadata.get('chunk_id')
            filename = doc.metadata.get('filename')
            anchor = f"chunk-{chunk_id}"
            st.markdown(f"- [Chunk {chunk_id}](#{anchor}) — *{filename}*")

        st.markdown("#### 🧩 Full Source Chunks")
        for doc in reranked_docs:
            chunk_id = doc.metadata.get('chunk_id')
            filename = doc.metadata.get('filename')
            anchor = f"chunk-{chunk_id}"

            st.markdown(f"<a name='{anchor}'></a>", unsafe_allow_html=True)
            st.markdown(f"**[Chunk {chunk_id}]** — *{filename}*")
            st.code(doc.page_content.strip()[:1000])
