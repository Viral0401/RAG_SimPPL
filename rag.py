import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import nltk

nltk.download('averaged_perceptron_tagger_eng')

# --- Load .env ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Config ---
embedding_model = "text-embedding-3-small"
doc_dir = "Grants"
persist_dir = "./vectorstore"
chunk_size = 800
chunk_overlap = 200
top_k_initial = 15
top_k_final = 5

# --- Streamlit Config ---
st.set_page_config(page_title="Grant & Fellowship Chat Assistant", layout="wide")
st.title("💬 Grant & Fellowship Application Assistant")

# --- Chat Memory ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

# --- Document Loader ---
@st.cache_resource
def load_all_documents():
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for path in Path(doc_dir).rglob("*"):
        if path.suffix not in [".pdf", ".docx"]:
            continue
        loader = PyPDFLoader(str(path)) if path.suffix == ".pdf" else UnstructuredWordDocumentLoader(str(path))
        try:
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            for i, doc in enumerate(chunks):
                doc.metadata["filename"] = str(path.relative_to(doc_dir)).replace("\\", "/")
                doc.metadata["chunk_id"] = i
                all_chunks.append(doc)
        except Exception as e:
            st.warning(f"❌ Error loading {path}: {e}")
    return all_chunks

# --- Vectorstore ---
def get_or_create_vectorstore(chunks):
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
        embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(persist_dir)
    else:
        embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

# --- Rerank ---
def rerank_with_llm(question, docs, top_n=5):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
    scored = []
    for doc in docs:
        prompt = f"""You are a helpful assistant. Score how relevant this chunk is to answering the user's question.

        Chunk:
        \"\"\"
        {doc.page_content}
        \"\"\"

        Question: {question}

        Score (0 to 10):"""
        try:
            score = float(llm.predict(prompt).strip())
        except:
            score = 0.0
        scored.append((score, doc))
    return [doc for score, doc in sorted(scored, reverse=True, key=lambda x: x[0])[:top_n]]

# --- Main Chat Interaction ---
chunks = load_all_documents()
vectorstore = get_or_create_vectorstore(chunks)
# #---------------------
# all_docs = vectorstore.similarity_search("dummy", k=1000)
# filenames = set(doc.metadata.get("filename", "unknown") for doc in all_docs)

# st.markdown("### 📁 Documents in Vectorstore:")
# for fname in sorted(filenames):
#     st.markdown(f"- {fname}")
# #----------------------
query = st.chat_input("Ask something about your grant/fellowship...")

if query:
    with st.spinner("Thinking..."):
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k_initial})
        relevant_chunks = retriever.get_relevant_documents(query)
        reranked = rerank_with_llm(query, relevant_chunks, top_k_final)

        # Inline Citations
        context = ""
        inline_map = {}
        for doc in reranked:
            marker = f"[Chunk {doc.metadata['chunk_id']}]"
            inline_map[marker] = (doc.metadata['chunk_id'], doc.metadata['filename'], doc.page_content)
            context += f"{marker}\n{doc.page_content}\n\n---\n\n"

        # Construct LLM prompt using memory
        full_history = ""
        for turn in st.session_state.chat_history:
            full_history += f"User: {turn['question']}\nAssistant: {turn['answer' ]}\n\n"

        chat_prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant for the SimPPL organization supporting someone with fellowship and grant applications.

        Only use the **provided document context** to answer. Do NOT answer from memory or external knowledge.
        Use inline citations like [Chunk X] to indicate where each part of your answer comes from.

        If the context does not contain an exact match, look for related or adjacent examples (e.g. similar themes, past projects, or comparable goals).

        Here is the chat so far:
        {history}

        Now, the user has asked:
        Question: {question}

        Here is the relevant document context:
        {context}

        Give your answer using only the context, with inline citations like [Chunk X].
        """)
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model = "gpt-4o")
        #gpt-3.5-turbo gpt-4o o1 o3-mini
        prompt = chat_prompt.format(history=full_history, context=context, question=query)
        answer = llm.predict(prompt)

        # Save chat
        st.session_state.chat_history.append({"question": query, "answer": answer, "sources": inline_map})

# --- Display Chat Thread ---
for turn in st.session_state.chat_history:
    st.chat_message("user").markdown(turn["question"])
    st.chat_message("assistant").markdown(turn["answer"])
    with st.expander("📄 Source Chunks"):
        for marker, (cid, fname, content) in turn["sources"].items():
            st.markdown(f"**{marker}** — *{fname}*")
            st.code(content.strip()[:1000])
