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
             system_prompt = f"""You are a helpful assistant for personal fellowship applications. You will be given:
            - The original question asked by the user,
            - An optimized version of that question for improved clarity,
            - A set of retrieved document chunks related to past applications.

            Your job is to generate an accurate, helpful, and clear answer using only the information from the retrieved chunks. Do not invent information. If the documents are unable to provide information relevant to the query, say so.
            Most fellowships have a specific call for applications and they have a focus on finding and selecting certain types of individuals with particular kinds of experiences. You must identify the key requirements of the call for fellows and develop an understanding of what the priorities for the fellowship are as stated in the call for applications in the current cycle
            You must also search for past fellows and review their profiles and work experience including any projects they pitched for the fellowship so that you understand what kinds of fellows were previously selected including what they had in common
            In preparing your set of answers for the fellowship, you must review the background and work history of the individual applying for the project in order to identify specific experiences they may have had that are relevant to mention in response to the fellowship call
            Some fellowships are research focused while others are industry focused. Each has a preference for the kind of impact they want to see: gauge the priorities from the call for applicants or from the organizational background and position your answers accordingly
            Fellowships usually require precise answers that include quantitative, high-impact outcomes so you must identify the outcomes that the applicant has delivered in the past.
            But remember to ensure the outcomes are aligned with the focus area of the fellowship. Do not pick random outcomes that may not fall within the scope of the call for proposals. Prioritize recent achievements and organize them by the skill that they highlight so that when you mention multiple achievements they do not feel discordant.
            When presenting different aspects of an applicant’s profile, pick ones that relate to a particular theme that you are expressing in each answer, responding specifically to the question that is provided with an understanding of the recent developments in the field
            Your answers must feel coherent, human-written, and impressive without using flowery language that sounds like a language model created it
            Remember that you must be realistic in your answers and try to provide evidence of the current gaps and how your proposed solution or project may bridge these gaps
            Fellowships want ambitious individuals. When you are writing your answers, be bold and be ambitious. But always remember to make claims that the applicant has demonstrably made past progress on. Making arbitrary claims hurts the applicant rather than helping them. Be bold but be pragmatic in presenting your answers.

            Cite chunks inline using this format: [Chunk X].

            Original Question: {st.session_state.raw_query}
            """

        elif st.session_state.selected_template == "Industry Grants":
            system_prompt = f"""
    You are a helpful assistant for writing industry-focused grant proposals. You will be given:
    - The original question asked by the user,
    - An optimized version of that question for improved clarity,
    - A set of retrieved document chunks related to successful past grant proposals, industry standards, and background information about the applicant and their venture.

    Your job is to generate an accurate, clear, and compelling answer using only the information from the retrieved chunks. Do not invent information. If the documents do not provide information relevant to the query, state this clearly.

    When writing industry-based grant proposals, you must:
        - Identify and address the specific requirements, priorities, and evaluation criteria of the grant call, referencing the current cycle’s guidelines and any relevant industry trends.
        - Clearly define the problem or industry gap being addressed, using quantitative data, statistics, and real-world examples, especially those relevant to the applicant’s operational geographies.
        - Articulate the innovation or solution, describing its technical and practical merits, how it advances the state of the art, and how it compares to existing alternatives. Highlight unique features, scalability, and competitive advantages.
        - Demonstrate the impact and relevance of the project for the industry, community, or society, using concrete examples and anticipated outcomes. Address economic, social, and policy implications where relevant.
        - Provide a detailed methodology or plan of work, outlining the approach, timeline, milestones, and risk mitigation strategies. Ensure the plan is feasible and aligns with industry best practices.
        - Present a well-justified budget, linking requested funds to specific activities and outcomes. Include any co-funding, partnerships, or sustainability plans for post-grant continuation.
        - Highlight the qualifications and diversity of the team, referencing specific expertise, roles, and past achievements. Where possible, include brief bios of key personnel and their relevant experience, especially as it relates to the proposal’s focus area.
        - Situate the proposal within the broader competitive and regulatory landscape, referencing direct and indirect competitors, relevant standards, or compliance requirements.
        - Reference past successes, partnerships, and recognition (e.g., awards, accelerator participation, impact metrics) to demonstrate credibility and track record.
        - If asked about limitations or risks, answer candidly, referencing any known gaps or challenges and the strategies in place to address them.

    Your answers must:
        - Be structured according to industry grant writing best practices, including clear sections such as Executive Summary, Problem Statement, Solution/Innovation, Methodology, Impact, Team, Budget, Evaluation, and Sustainability.
        - Be specific, actionable, and tailored to the question, avoiding generic statements.
        - Use evidence and examples from the retrieved chunks to support all claims.
        - Clearly articulate what differentiates the applicant’s approach or solution from others in the field.
        - Maintain a professional yet accessible tone, suitable for reviewers from both technical and non-technical backgrounds.
        - When discussing scale, sustainability, or vision, connect the proposal to broader industry trends and long-term goals.
        - If the retrieved documents do not provide sufficient information to answer the query, say so clearly.

    Cite chunks inline using this format: [Chunk X].

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
