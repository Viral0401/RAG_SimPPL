import os
import sys
import types
from pathlib import Path
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
import dspy  # type: ignore
from dspy import Prediction  # type: ignore

from grant_cot_module import GrantCoTModule
optimized_cot = GrantCoTModule()
optimized_cot.load("optimized_grant_cot.json")

# --- LiteLLM Patch ---
os.environ["LITELLM_USE_CACHE"] = "False"
os.environ["LITELLM_LOGGING"] = "False"
dummy_module = types.SimpleNamespace()
dummy_module.TranscriptionCreateParams = types.SimpleNamespace()
dummy_module.TranscriptionCreateParams.__annotations__ = {}
sys.modules["litellm.litellm_core_utils.openai_types"] = dummy_module

# --- API Key ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# --- DSPy Init ---
lm = dspy.LM("openai/gpt-4o", api_key=OPENAI_API_KEY)
try:
    dspy.configure(lm=lm)
except RuntimeError:
    pass

# --- Streamlit Setup ---
st.set_page_config(page_title="Grant & Fellowship Chat Assistant", layout="wide")
st.title("💬 Grant & Fellowship Application Assistant")

# --- Optional Sidebar Context ---
with st.sidebar:
    st.markdown("### 📝 Grant/Fellowship Context (Optional)")
    user_context = st.text_area("Add any specific details about the grant you're applying for.", key="grant_context_sidebar")

# --- Clear Chat ---
# --- Clear Chat (inline, single-line layout) ---
st.markdown(
    """
    <div style="display: flex; justify-content: flex-end; margin-top: -2rem;">
        <form action="" method="post">
            <button style="background-color: #1c1c1c; color: #fff; padding: 0.4rem 0.8rem; border: 1px solid #555; border-radius: 6px; cursor: pointer;" type="submit" name="clear">🧹 Clear Chat</button>
        </form>
    </div>
    """,
    unsafe_allow_html=True
)

# Clear state if button clicked
if "clear" in st.query_params:
    st.session_state.clear()
    st.rerun()


# --- Defaults ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_template" not in st.session_state:
    st.session_state.selected_template = "Personal Fellowships"

# --- Config ---
embedding_model = "text-embedding-3-small"
doc_dir = "Grants"
persist_dir = "./vectorstore"
chunk_size = 1000
chunk_overlap = 200
top_k_vector = 15
top_k_bm25 = 3
top_k_final = 5

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

chunks = load_all_documents()
vectorstore = get_or_create_vectorstore(chunks)

# --- DSPy Modules ---
class QueryOptimizationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.optimize = dspy.Predict("query -> optimized_query")

    def forward(self, query):
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
        prompt = (
            "Rewrite the following grant application question to be as clear, specific, and context-rich as possible for searching a knowledge base. "
            "Do not add unrelated information. Only return the improved query.\n\n"
            f"Original question: {query}\n\nOptimized query:"
        )
        return llm.invoke(prompt).content.strip()

class HybridRetrievalModule(dspy.Module):
    def __init__(self, vectorstore, all_docs):
        super().__init__()
        self.vectorstore = vectorstore
        self.all_docs = all_docs

    def forward(self, optimized_query, top_k_vector=15, top_k_bm25=5):
        vector_results = self.vectorstore.similarity_search(optimized_query, k=top_k_vector)
        bm25_retriever = BM25Retriever.from_documents(self.all_docs)
        bm25_results = bm25_retriever.invoke(optimized_query)[:top_k_bm25]
        seen, combined = set(), []
        for doc in vector_results + bm25_results:
            doc_id = (doc.metadata.get("filename"), doc.metadata.get("chunk_id"))
            if doc_id not in seen:
                combined.append(doc)
                seen.add(doc_id)
        return combined

class RerankModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rerank = dspy.Predict("question, docs -> top_docs")

    def forward(self, question, docs, top_n=5):
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
        scored = []
        for doc in docs:
            prompt = f"""You are a helpful assistant. Score how relevant this chunk is to answering the user's question.
Chunk:
\"\"\"{doc.page_content}\"\"\"
Question: {question}
Score (0 to 10):"""
            try:
                score = float(llm.predict(prompt).strip())
            except:
                score = 0.0
            scored.append((score, doc))
        return [doc for score, doc in sorted(scored, reverse=True, key=lambda x: x[0])[:top_n]]

class GrantRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict("system_prompt, context, question, history -> answer")

    def forward(self, system_prompt, context, question, history):
        return self.generate(system_prompt=system_prompt, context=context, question=question, history=history)

# --- Template Buttons ---
st.subheader("🧠 Choose Grant Type Template")
col1, col2, col3 = st.columns(3)
if col1.button("🎓 Personal Fellowships"):
    st.session_state.selected_template = "Personal Fellowships"
if col2.button("🏭 Industry Grants"):
    st.session_state.selected_template = "Industry Grants"
if col3.button("📚 Academic Research Grants"):
    st.session_state.selected_template = "Academic Research Grants"
st.markdown(f"📌 **Selected Template:** `{st.session_state.selected_template}`")

# --- Query Input ---
query = st.chat_input("Ask something about your grant/fellowship...")

if query:
    with st.spinner("🔍 Optimizing Query..."):
        query_optimizer = QueryOptimizationModule()
        optimized_query = query_optimizer(query)
        st.session_state.optimized_query = optimized_query
        st.session_state.raw_query = query
        st.session_state.proceed_triggered = False

if "optimized_query" in st.session_state and not st.session_state.get("proceed_triggered", False):
    st.markdown("### ✍️ Optimized Query (editable)")
    edited_query = st.text_area("Modify the optimized query below if needed:", st.session_state.optimized_query)
    if st.button("✅ Proceed with this optimized query"):
        st.session_state.optimized_query = edited_query.strip()
        st.session_state.proceed_triggered = True
        st.rerun()

# --- Retrieval + Answer ---
if st.session_state.get("proceed_triggered", False):
    with st.spinner("🔍 Retrieving + Generating Answer..."):
        retriever = HybridRetrievalModule(vectorstore, chunks)
        initial_docs = retriever(st.session_state.optimized_query, top_k_vector=top_k_vector, top_k_bm25=top_k_bm25)
        reranker = RerankModule()
        top_docs = reranker(st.session_state.optimized_query, initial_docs, top_n=top_k_final)

        context, inline_map = "", {}
        for doc in top_docs:
            cid = doc.metadata["chunk_id"]
            marker = f"[Chunk {cid}](#chunk-{cid})"
            context += f"{marker}\n{doc.page_content}\n\n---\n\n"
            inline_map[marker] = (cid, doc.metadata["filename"], doc.page_content)

        full_history = ""
        for turn in st.session_state.chat_history:
            full_history += f"User: {turn['question']}\nAssistant: {turn['answer']}\n\n"

        # --- System Prompt ---
        if st.session_state.selected_template == "Personal Fellowships":
            system_prompt = f"""You are a helpful assistant for personal fellowship applications such as Mozilla Fellowships or similar. You will be given:
- The original question asked by the user,
- An optimized version of that question for improved clarity,
- A set of retrieved document chunks related to past applications.

Your job is to generate an accurate, helpful, and clear answer using only the information from the retrieved chunks. Do not invent information. If the documents don’t cover the query, say so.

Cite chunks inline using this format: [Chunk X].

Original Question: {st.session_state.raw_query}
"""
        elif st.session_state.selected_template == "Industry Grants":
            system_prompt = f"""You are a knowledgeable assistant for industry grant applications, including those from programs like exploreCSR or corporate tech initiatives. You will receive:
- The user's original question,
- An optimized version of that question,
- Relevant document chunks from previous successful applications.

Use only the document content to answer. Don't hallucinate or generalize. Cite relevant chunks inline like [Chunk X]. If context is missing, explicitly state that.

Original Question: {st.session_state.raw_query}
"""
        else:
            system_prompt = f"""You are an expert assistant for academic research grants, such as MIT PKG Innovation Fellowships or university-funded research initiatives. You will be provided:
- The original user-submitted question,
- An optimized version for better retrieval,
- Contextual chunks from prior grant documents.

Craft a detailed response using only the provided chunked context. Support your answer with citations like [Chunk X]. Do not answer beyond what the retrieved content supports.

Original Question: {st.session_state.raw_query}
"""

        if user_context.strip():
            system_prompt += f"\n\nAdditional context about this grant: {user_context.strip()}"

        cot_result = optimized_cot(question=st.session_state.optimized_query)
        answer = cot_result.answer
        reasoning = cot_result.reasoning

        st.session_state.chat_history.append({
            "question": st.session_state.raw_query,
            "optimized_query": st.session_state.optimized_query,
            "reasoning": reasoning,
            "answer": answer,
            "sources": inline_map
        })
        st.session_state.proceed_triggered = False

# --- Chat History ---
if st.session_state.get("chat_history"):
    for turn in st.session_state.chat_history:
        st.chat_message("user").markdown(turn["question"])
        st.chat_message("assistant").markdown(turn.get("answer", ""))
        if "reasoning" in turn:
            st.markdown(f"**🧠 Reasoning:** {turn['reasoning']}")
        st.markdown(f"**Optimized Query Used:** {turn.get('optimized_query', '')}")
        with st.expander("📄 Source Chunks"):
            for marker, (cid, fname, content) in turn["sources"].items():
                anchor = f"chunk-{cid}"
                st.markdown(f"<a name='{anchor}'></a>", unsafe_allow_html=True)
                st.markdown(f"**{marker}** — *{fname}*")
                st.code(content.strip())
