
import streamlit as st
import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PharmaRegAI",
    page_icon="🧪",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* =========================================================
   GLOBAL
========================================================= */

.stApp {
    background-color: #F5F7F8;
    color: #111827;
}

.main {
    background-color: #F5F7F8;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}


/* =========================================================
   HERO TITLE
========================================================= */

.hero-title {
    font-size: 44px;
    font-weight: 800;
    color: #071A21;
    margin-bottom: 6px;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 17px;
    font-weight: 500;
    color: #37474F;
    margin-bottom: 28px;
}


/* =========================================================
   SECTION HEADINGS
========================================================= */

.section-title {
    font-size: 23px;
    font-weight: 750;
    color: #071A21;
    margin-top: 28px;
    margin-bottom: 14px;
}


/* =========================================================
   ANSWER CARD
========================================================= */

.answer-card {
    background-color: #FFFFFF;
    color: #111827;

    padding: 26px;

    border-radius: 14px;

    border: 1px solid #CBD5D9;

    box-shadow: 0 4px 16px rgba(7, 26, 33, 0.08);

    margin-top: 15px;

    font-size: 16px;
    line-height: 1.7;
}


/* =========================================================
   SOURCE CARD
========================================================= */

.source-card {
    background-color: #FFFFFF;
    color: #111827;

    padding: 18px;

    border-radius: 12px;

    border: 1px solid #CBD5D9;

    box-shadow: 0 2px 8px rgba(7, 26, 33, 0.06);

    margin-bottom: 12px;

    font-size: 14px;
    line-height: 1.6;
}

.source-card strong {
    color: #071A21;
}


/* =========================================================
   SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {
    background-color: #071A21;
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}


/* =========================================================
   TEXT INPUT
========================================================= */

textarea {
    background-color: #FFFFFF !important;
    color: #111827 !important;

    border: 2px solid #B8C4C8 !important;

    border-radius: 10px !important;
}

textarea:focus {
    border-color: #087F8C !important;
    box-shadow: 0 0 0 2px rgba(8, 127, 140, 0.15) !important;
}


/* =========================================================
   BUTTONS
========================================================= */

.stButton > button {
    background-color: #087F8C;
    color: #FFFFFF;

    border: none;

    border-radius: 9px;

    font-weight: 650;

    padding: 10px 18px;

    transition: all 0.2s ease;
}

.stButton > button:hover {
    background-color: #065F69;
    color: #FFFFFF;
    border: none;
}


/* =========================================================
   PRIMARY ASK BUTTON
========================================================= */

.stButton > button[kind="primary"] {
    background-color: #071A21;
    color: #FFFFFF;

    font-weight: 700;

    border-radius: 10px;
}

.stButton > button[kind="primary"]:hover {
    background-color: #087F8C;
    color: #FFFFFF;
}


/* =========================================================
   DISCLAIMER
========================================================= */

.disclaimer {
    background-color: #FFF7D6;

    border: 1px solid #E6C94C;

    padding: 16px;

    border-radius: 10px;

    color: #3D3500;

    font-size: 14px;

    line-height: 1.6;

    margin-top: 30px;
}


/* =========================================================
   FOOTER
========================================================= */

.footer {
    text-align: center;

    color: #4B5A60;

    font-size: 13px;

    margin-top: 40px;

    padding-top: 20px;

    border-top: 1px solid #CBD5D9;
}


/* =========================================================
   DIVIDERS
========================================================= */

hr {
    border-color: #CBD5D9 !important;
}


/* =========================================================
   SPINNER
========================================================= */

.stSpinner > div {
    border-top-color: #087F8C !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="hero-title">🧪 PharmaRegAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-subtitle">
    Retrieval-Augmented Regulatory Intelligence System for
    Pharmaceutical Supply Chains
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("About PharmaRegAI")

    st.write(
        """
        PharmaRegAI retrieves information from pharmaceutical
        regulatory documents and generates answers using
        retrieval-augmented generation (RAG).
        """
    )

    st.divider()

    st.subheader("Technology")

    st.write("• Sentence Transformers")
    st.write("• FAISS Vector Search")
    st.write("• FLAN-T5")
    st.write("• LangChain")
    st.write("• Streamlit")

    st.divider()

    st.subheader("Document Sources")

    st.write("DRAP")
    st.write("WHO")
    st.write("Pharmaceutical Guidelines")

    st.divider()

    st.caption(
        "Developed as a Data Science / AI research project."
    )


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================================
# LOAD VECTOR DATABASE
# =========================================================

@st.cache_resource
def load_vectorstore():

    embeddings = load_embeddings()

    db = FAISS.load_local(
        "pharmareg_faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


# =========================================================
# LOAD FLAN-T5
# =========================================================

@st.cache_resource
def load_model():

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    return tokenizer, model


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(prompt):

    tokenizer, model = load_model()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer


# =========================================================
# CREATE RAG PROMPT
# =========================================================

def create_rag_prompt(question, retrieved_docs):

    context = ""

    for i, doc in enumerate(retrieved_docs, 1):

        context += f"""
SOURCE {i}

Document:
{doc["source"]}

Page:
{doc["page"]}

Content:
{doc["text"]}

-----------------------------------
"""

    prompt = f"""
You are PharmaRegAI, a pharmaceutical regulatory
information assistant.

Answer the question using ONLY the regulatory
documents provided below.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the documents do not contain enough information,
   say that the available documents do not provide
   sufficient information.
4. Give a clear and concise answer.
5. Mention the document name and page number when possible.

REGULATORY DOCUMENTS:

{context}

QUESTION:

{question}

ANSWER:
"""

    return prompt


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_documents(query, k=5):

    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search_with_score(
        query,
        k=k
    )

    retrieved = []

    for doc, score in results:

        retrieved.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source_file"),
            "page": doc.metadata.get("page"),
            "score": float(score)
        })

    return retrieved


# =========================================================
# QUESTION INPUT
# =========================================================

st.markdown(
    '<div class="section-title">Ask a Regulatory Question</div>',
    unsafe_allow_html=True
)

question = st.text_area(
    "Enter your question",
    placeholder="Example: How should temperature-sensitive pharmaceutical products be handled?",
    height=110
)


# =========================================================
# EXAMPLE QUESTIONS
# =========================================================

st.markdown("**Example questions:**")

example_questions = [
    "What is pharmacovigilance?",
    "How should biological products be transported?",
    "What is a drug recall?",
    "What are the requirements for medicine storage?"
]

cols = st.columns(4)

for i, example in enumerate(example_questions):

    with cols[i]:

        if st.button(
            example,
            key=f"example_{i}",
            use_container_width=True
        ):
            question = example


# =========================================================
# ASK BUTTON
# =========================================================

if st.button(
    "🔍 Ask PharmaRegAI",
    type="primary",
    use_container_width=True
):

    if not question.strip():

        st.warning("Please enter a regulatory question.")

    else:

        with st.spinner(
            "Searching regulatory documents and generating answer..."
        ):

            retrieved = retrieve_documents(
                question,
                k=5
            )

            prompt = create_rag_prompt(
                question,
                retrieved
            )

            answer = generate_answer(prompt)


        # =================================================
        # ANSWER
        # =================================================

        st.markdown(
            '<div class="section-title">🤖 Answer</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="answer-card">
            {answer}
            </div>
            """,
            unsafe_allow_html=True
        )


        # =================================================
        # SOURCES
        # =================================================

        st.markdown(
            '<div class="section-title">📚 Retrieved Sources</div>',
            unsafe_allow_html=True
        )

        for i, doc in enumerate(retrieved[:3], 1):

            st.markdown(
                f"""
                <div class="source-card">

                <strong>Source {i}</strong><br><br>

                📄 <strong>Document:</strong>
                {doc["source"]}<br>

                📑 <strong>Page:</strong>
                {doc["page"]}<br>

                🔎 <strong>Retrieval distance:</strong>
                {doc["score"]:.4f}

                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown(
    """
    <div class="disclaimer">

    ⚠️ <strong>Important:</strong>
    PharmaRegAI is an academic/research prototype.
    It provides information retrieved from the included
    regulatory documents and should not replace official
    regulatory guidance or professional pharmaceutical advice.

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
    PharmaRegAI • Retrieval-Augmented Regulatory Intelligence
    </div>
    """,
    unsafe_allow_html=True
)
