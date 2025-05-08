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
import base64
from io import BytesIO
import fitz
import requests
import json
import openai
from openai import OpenAI
from mistralai import Mistral # type: ignore

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
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]

# --- DSPy Init ---
lm = dspy.LM("openai/gpt-4o", api_key=OPENAI_API_KEY)
try:
    dspy.configure(lm=lm)
except RuntimeError:
    pass

def fetch_with_perplexity(query, api_key):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": query}],
        "search": True
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"🌐 Perplexity API error: {e}"

def refine_answer_with_instruction(system_prompt, context, original_question, optimized_question, previous_answer, user_instruction):
    refine_prompt = f"""
        {system_prompt}
        Below is the context and the original exchange. The user now wants to refine the answer with additional instructions.

        Context:
        {context}

        Original Question: {original_question}
        Optimized Question: {optimized_question}

        Previous Answer:
        {previous_answer}

        User Instruction:
        {user_instruction}

        Refined Answer:
        """
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": refine_prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def extract_text_with_mistral_ocr(uploaded_pdf):
    # Read the uploaded PDF file
    pdf_bytes = uploaded_pdf.read()
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

    # Initialize Mistral client
    api_key = MISTRAL_API_KEY
    client = Mistral(api_key=api_key)

    # Process the PDF with Mistral OCR
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{base64_pdf}"
        },
        include_image_base64=False  # Set to True if you want images included
    )

    # Concatenate markdown content from all pages
    extracted_text = "\n\n".join(page.markdown for page in ocr_response.pages)
    return extracted_text


def extract_questions_with_gpt4o(document_text, api_key):
    prompt = f"""
Extract all questions from the following grant or fellowship document text. Return ONLY the questions as a JSON list.

Text:
\"\"\"
{document_text[:30000]}
\"\"\"

Questions:
"""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = response.choices[0].message.content
    import json, re
    try:
        questions = json.loads(content)
    except Exception:
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            questions = json.loads(match.group(0))
        else:
            questions = []
    return questions

# --- Streamlit Setup ---
st.set_page_config(page_title="Grant & Fellowship Chat Assistant", layout="wide")
st.title("💬 Grant & Fellowship Application Assistant")

# --- Optional Sidebar Context ---
with st.sidebar:
    st.markdown("### 📥 Upload Grant/Reference Documents (Will Be Added to Vectorstore)")
    new_grant_file = st.file_uploader(
        "Upload a new PDF or DOCX to be added to the Grants folder and vectorstore:",
        type=["pdf", "docx"],
        key="grant_uploader"
    )
    # st.markdown("### 📝 Grant/Fellowship Context (Optional)")
    # user_context = st.text_area(
    #     "Add any specific details about the grant you're applying for.",
    #     key="grant_context_sidebar"
    # )
    # st.markdown("---")
    st.markdown("---")

    # 🆕 Combined Grant URL and Grant Name Input
    st.markdown("#### 🌐 Grant Website and Name (Combined)")
    grant_combined_input = st.text_area(
        "Paste the grant/fellowship website link and/or exact name:",
        help="Example:\nhttps://gatesfoundation.org/grants\nGates Foundation Grand Challenges 2025"
    )

    st.markdown("---")

    # File uploader (sidebar)
    uploaded_pdf = st.file_uploader(
        "📂 Upload a Grant PDF",
        type=["pdf"],
        label_visibility="visible"
    )

    pdf_text = ""
    if uploaded_pdf:
        # Show PDF viewer
        base64_pdf = base64.b64encode(uploaded_pdf.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400px" type="application/pdf"></iframe>'
        st.markdown("### 📄 PDF Preview")
        st.markdown(pdf_display, unsafe_allow_html=True)

        # --- Mistral OCR Extraction ---
        with st.spinner("🔍 Extracting text from PDF using Mistral OCR..."):
            api_key = st.secrets["MISTRAL_API_KEY"]
            client = Mistral(api_key=api_key)

            uploaded_pdf.seek(0)
            pdf_base64 = base64.b64encode(uploaded_pdf.read()).decode("utf-8")

            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{pdf_base64}"
                },
                include_image_base64=False
            )

            pdf_text = "\n\n".join(page.markdown for page in ocr_response.pages)

        # --- Question Extraction with GPT-4o ---
        if pdf_text and "extracted_questions" not in st.session_state:
            with st.spinner("🔍 Extracting questions from document..."):
                questions = extract_questions_with_gpt4o(
                    pdf_text, OPENAI_API_KEY
                )
                st.session_state.extracted_questions = questions

        if st.session_state.get("extracted_questions"):
            st.markdown("### 📋 Extracted Questions")
            for q in st.session_state.extracted_questions:
                if st.button(q[:75] + "..." if len(q) > 75 else q, key=f"q_{hash(q)}"):
                    st.session_state.raw_query = q
                    st.rerun()
                    st.session_state.input_triggered_by_click = True

# --- Fetch Grant Information from Perplexity ---
webpage_text = ""
if grant_combined_input.strip():
    with st.spinner("🌐 Fetching Grant Info via Perplexity..."):
        try:
            browse_query = f"""
            You are a researcher. Please extract all important information from the provided URL (if present) and the grant name.

            Input:
            {grant_combined_input}

            Return a clean detailed structured text.
            """
            webpage_text = fetch_with_perplexity(browse_query, st.secrets["PERPLEXITY_API_KEY"])
        except Exception as e:
            st.warning(f"🌐 Could not fetch grant data: {e}")

    # if webpage_text.strip():
    #     st.markdown("### 🌐 Grant/Fellowship Information Retrieved from the Web")
    #     st.info(webpage_text)

col1, col2, col3 = st.columns([7, 1, 1])
with col3:
    if st.button("🧹 Clear Chat"):
        st.session_state.clear()
        st.rerun()

# Handle clear chat logic
if st.session_state.get("clear_chat"):
    st.session_state.clear()
    st.rerun()

# --- Defaults ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "latest_system_prompt" not in st.session_state:
    st.session_state.latest_system_prompt = ""

if "latest_context" not in st.session_state:
    st.session_state.latest_context = ""

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

# --- Process Uploaded Grant Document and Add to Vectorstore ---
if new_grant_file:
    file_path = Path("Grants") / new_grant_file.name
    with open(file_path, "wb") as f:
        f.write(new_grant_file.read())
    st.success(f"✅ Saved `{new_grant_file.name}` to Grants/")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    loader = PyPDFLoader(str(file_path)) if file_path.suffix == ".pdf" else UnstructuredWordDocumentLoader(str(file_path))

    try:
        docs = loader.load()
        new_chunks = splitter.split_documents(docs)
        for i, doc in enumerate(new_chunks):
            doc.metadata["filename"] = str(file_path.relative_to("Grants")).replace("\\", "/")
            doc.metadata["chunk_id"] = i

        embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=OPENAI_API_KEY)
        vectorstore.add_documents(new_chunks)
        vectorstore.save_local(persist_dir)
        st.success("✅ Document successfully embedded and added to vectorstore!")
    except Exception as e:
        st.error(f"❌ Failed to process document: {e}")

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

if st.session_state.get("input_triggered_by_click") and not query:
    query = st.session_state.raw_query
    st.session_state.input_triggered_by_click = False  # Reset

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

        # if user_context.strip():
        #     system_prompt += f"\n\nAdditional context about this grant: {user_context.strip()}"

        if pdf_text.strip():
            system_prompt += "\n\n---\n📄 Uploaded Grant PDF Content:\n" + pdf_text.strip()

        if webpage_text.strip():
            system_prompt += "\n\n---\n🌐 Retrieved Grant/Fellowship Info from Website:\n" + webpage_text.strip()

        # Store for use in Edit Answer later
        st.session_state.latest_system_prompt = system_prompt
        st.session_state.latest_context = context


        cot_result = optimized_cot(question=st.session_state.optimized_query)
        answer = cot_result.answer
        reasoning = cot_result.reasoning


        # --- Internet (Perplexity) Fetch and Answer Generation ---
        pplx_answer = None
        pplx_generated_answer = None
        if st.session_state.selected_template == "Academic Research Grants":
            with st.spinner("🌐 Fetching Internet Context via Perplexity..."):
                pplx_answer = fetch_with_perplexity(st.session_state.raw_query, st.secrets["PERPLEXITY_API_KEY"])

            if pplx_answer:
                web_prompt = f"""
                You are an academic research assistant. Based ONLY on the following Internet information, answer the user's question clearly and accurately.

                --- Internet Context ---
                {pplx_answer}
                ------------------------

                User's Question: {st.session_state.raw_query}

                Answer:
                """
                web_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.3)
                pplx_generated_answer = web_llm.invoke(web_prompt).content.strip()

        # --- Store everything in session ---
        history_item = {
            "question": st.session_state.raw_query,
            "optimized_query": st.session_state.optimized_query,
            "reasoning": reasoning,
            "answer": answer,
            "sources": inline_map,
            "system_prompt": system_prompt,
            "context": context
        }
        if pplx_answer:
            history_item["pplx_answer"] = pplx_answer  # raw web search
        if pplx_generated_answer:
            history_item["pplx_generated_answer"] = pplx_generated_answer  # full generated answer from web

        st.session_state.chat_history.append(history_item)
        st.session_state.proceed_triggered = False

# --- Chat History ---
if st.session_state.get("chat_history"):
    for turn in st.session_state.chat_history:
        st.chat_message("user").markdown(turn["question"])
        st.chat_message("assistant").markdown(turn.get("answer", ""))

        with st.expander("✏️ Edit Answer"):
            edit_instruction = st.text_area(
                f"Provide edits or additions for this answer:",
                key=f"instruction_{hash(turn['question'])}"
            )
            if st.button("🔄 Apply Instruction", key=f"apply_{hash(turn['question'])}"):
                with st.spinner("✍️ Refining answer..."):
                    refined = refine_answer_with_instruction(
                    system_prompt=st.session_state.get("latest_system_prompt", ""),
                    context=st.session_state.get("latest_context", ""),
                    original_question=turn['question'],
                    optimized_question=turn['optimized_query'],
                    previous_answer=turn['answer'],
                    user_instruction=edit_instruction
                    )
                    turn['answer'] = refined
                    st.rerun()

        # Show separate Perplexity Internet-generated Assistant answer
        if "pplx_generated_answer" in turn:
            st.chat_message("assistant").markdown("🌐 **Answer based on Internet Search (Perplexity):**")
            st.chat_message("assistant").markdown(turn["pplx_generated_answer"])

        # Optional: show raw Internet context separately
        if "pplx_answer" in turn:
            with st.expander("🌐 Internet Context Summary (Perplexity)"):
                st.markdown(turn["pplx_answer"])

        if "reasoning" in turn:
            st.markdown(f"**🧠 Reasoning:** {turn['reasoning']}")

        st.markdown(f"**Optimized Query Used:** {turn.get('optimized_query', '')}")

        with st.expander("📄 Source Chunks"):
            for marker, (cid, fname, content) in turn["sources"].items():
                anchor = f"chunk-{cid}"
                st.markdown(f"<a name='{anchor}'></a>", unsafe_allow_html=True)
                st.markdown(f"**{marker}** — *{fname}*")
                st.code(content.strip())