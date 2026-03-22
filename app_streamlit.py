"""
app_streamlit.py — Document Intelligence System Demo
Combines:
  - PDF ingestion + FAISS indexing (existing pipeline)
  - RAG Q&A: user asks questions, answered from document context via Claude API

Run from project root:
    streamlit run app_streamlit.py
"""

import tempfile
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Document Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e6e0; }
[data-testid="stSidebar"] { background: #0f0f18 !important; border-right: 1px solid #1e1e2e; }

.doc-header { padding: 2rem 0 1.2rem 0; border-bottom: 1px solid #1e1e2e; margin-bottom: 1.5rem; }
.doc-header h1 { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.2rem; color: #e8e6e0; letter-spacing: -0.03em; margin: 0; }
.doc-header p { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #4a4a6a; margin: 0.3rem 0 0 0; letter-spacing: 0.08em; text-transform: uppercase; }
.accent { color: #7c6af7; }

.metric-row { display: flex; gap: 0.8rem; margin-bottom: 1.2rem; }
.metric-card { flex: 1; background: #0f0f18; border: 1px solid #1e1e2e; border-radius: 10px; padding: 1rem; text-align: center; }
.metric-value { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.8rem; color: #7c6af7; line-height: 1; }
.metric-label { font-family: 'DM Mono', monospace; font-size: 0.6rem; color: #4a4a6a; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem; }

/* Chat messages */
.msg-user {
    background: #13131f;
    border: 1px solid #2a2a3e;
    border-radius: 12px 12px 4px 12px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    font-family: 'Syne', sans-serif;
    font-size: 0.92rem;
    color: #e8e6e0;
    margin-left: 15%;
}
.msg-assistant {
    background: #0f0f18;
    border: 1px solid #1e1e2e;
    border-left: 3px solid #7c6af7;
    border-radius: 4px 12px 12px 12px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.83rem;
    color: #c8c6c0;
    line-height: 1.7;
    margin-right: 15%;
}
.msg-source {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #3a3a5a;
    margin-top: 0.6rem;
    padding-top: 0.5rem;
    border-top: 1px solid #1e1e2e;
}
.source-chip {
    display: inline-block;
    background: #13131f;
    border: 1px solid #2a2a3e;
    border-radius: 4px;
    padding: 0.1rem 0.5rem;
    margin-right: 0.3rem;
    color: #7c6af7;
    font-size: 0.62rem;
}

.stButton > button { background: #7c6af7 !important; color: white !important; border: none !important; border-radius: 8px !important; font-family: 'Syne', sans-serif !important; font-weight: 600 !important; padding: 0.5rem 1.5rem !important; }
.stButton > button:hover { opacity: 0.85 !important; }
hr { border-color: #1e1e2e !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Suggestion chips */
.suggestion { 
    display: inline-block; 
    background: #0f0f18; 
    border: 1px solid #2a2a3e; 
    border-radius: 20px; 
    padding: 0.3rem 0.9rem; 
    margin: 0.2rem; 
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem; 
    color: #7c6af7; 
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)


# ── Pipeline loader ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    from src.pipeline import DocumentIntelligencePipeline
    return DocumentIntelligencePipeline.from_config("configs/config.yaml")


# ── RAG answer function ────────────────────────────────────────────────────
def get_answer(question: str, pipeline, top_k: int = 5) -> dict:
    """
    Retrieve relevant chunks from FAISS and answer using Claude API.
    Returns dict with answer text and source chunks.
    """
    import groq
    import os

    # Step 1: Retrieve relevant chunks
    hits = pipeline.search(question, top_k=top_k)

    if not hits:
        return {
            "answer": "No relevant content found. Please upload and index a PDF first.",
            "sources": [],
        }

    # Step 2: Build context from top chunks
    context_parts = []
    for i, hit in enumerate(hits):
        context_parts.append(
            f"[Chunk {i+1} | {hit.chunk.pdf_name} | page {hit.chunk.page + 1}]\n{hit.chunk.text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Step 3: Call Claude API
    client = groq.Groq(api_key=st.session_state.get("api_key", ""))

    system_prompt = """You are a precise document assistant. Answer questions using ONLY the provided document chunks.
Rules:
- Answer directly and concisely based on the context
- If the answer isn't in the context, say so clearly  
- Quote specific details when relevant
- Do not add information beyond what's in the chunks"""

    user_prompt = f"""Document context:
{context}

Question: {question}

Answer based only on the above context:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1000,
        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
    )

    answer = response.choices[0].message.content

    sources = [
        {
            "text": hit.chunk.text[:120] + "..." if len(hit.chunk.text) > 120 else hit.chunk.text,
            "pdf": hit.chunk.pdf_name,
            "page": hit.chunk.page + 1,
            "score": round(hit.score, 3),
        }
        for hit in hits[:3]   # show top 3 sources
    ]

    return {"answer": answer, "sources": sources}


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem 0;">
        <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.1rem;color:#e8e6e0;">🔍 DocIntel</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#4a4a6a;text-transform:uppercase;letter-spacing:0.1em;margin-top:0.2rem;">Document Intelligence System</div>
    </div><hr>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#4a4a6a;margin-bottom:0.5rem;">
        ANTHROPIC API KEY
    </div>
    """, unsafe_allow_html=True)

    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="gsk_...",
        label_visibility="collapsed",
        help="Get your key at console.anthropic.com"
    )
    if api_key:
        st.session_state["api_key"] = api_key

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#3a3a5a;line-height:2;">
        <div>EXTRACTION</div><div style="color:#7c6af7;">LayoutLMv3</div>
        <div style="margin-top:0.4rem;">RETRIEVAL</div><div style="color:#7c6af7;">FAISS + MiniLM</div>
        <div style="margin-top:0.4rem;">GENERATION</div><div style="color:#7c6af7;">Claude Sonnet</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Indexed docs info
    if "indexed_docs" in st.session_state and st.session_state["indexed_docs"]:
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#4a4a6a;margin-bottom:0.4rem;">
            INDEXED DOCUMENTS
        </div>
        """, unsafe_allow_html=True)
        for doc in st.session_state["indexed_docs"]:
            st.markdown(f"""
            <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#7c6af7;
                        padding:0.3rem 0;border-bottom:1px solid #1e1e2e;">
                📄 {doc}
            </div>
            """, unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="doc-header">
    <h1>Document <span class="accent">Intelligence</span></h1>
    <p>Upload PDFs · Ask questions · Get answers grounded in your documents</p>
</div>
""", unsafe_allow_html=True)


# ── Two column layout ──────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.8], gap="large")


# ════════════════════════════════════════
# LEFT: Upload + Index
# ════════════════════════════════════════
with col_left:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;
                color:#e8e6e0;margin-bottom:0.8rem;">
        📄 Upload Document
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded:
        col_a, col_b = st.columns(2)
        with col_a:
            index_btn = st.button("Index Document", use_container_width=True)
        with col_b:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state["messages"] = []
                st.rerun()

        if index_btn:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)

            with st.spinner(f"Indexing {uploaded.name}..."):
                try:
                    pipeline = load_pipeline()
                    doc = pipeline.run(str(tmp_path))
                    doc.pdf_name = uploaded.name

                    if "indexed_docs" not in st.session_state:
                        st.session_state["indexed_docs"] = []
                    if uploaded.name not in st.session_state["indexed_docs"]:
                        st.session_state["indexed_docs"].append(uploaded.name)

                    fields = doc.fields
                    total_fields = sum(len(v) for v in fields.values())
                    st.session_state["last_stats"] = {
                        "pages": doc.pages,
                        "fields": total_fields,
                        "doc": uploaded.name,
                    }
                    st.success(f"Indexed! {doc.pages} pages, {total_fields} fields extracted.")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    tmp_path.unlink(missing_ok=True)

    # Stats
    if "last_stats" in st.session_state:
        s = st.session_state["last_stats"]
        st.markdown(f"""
        <div class="metric-row" style="margin-top:1rem;">
            <div class="metric-card">
                <div class="metric-value">{s['pages']}</div>
                <div class="metric-label">Pages</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{s['fields']}</div>
                <div class="metric-label">Fields</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Suggested questions
    if "indexed_docs" in st.session_state and st.session_state["indexed_docs"]:
        st.markdown("""
        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.9rem;
                    color:#e8e6e0;margin:1.2rem 0 0.6rem 0;">
            💡 Try asking
        </div>
        """, unsafe_allow_html=True)

        suggestions = [
            "What is the main topic of this document?",
            "Summarise the key points",
            "What are the eligibility criteria?",
            "What are the penalties mentioned?",
            "What forms need to be filed?",
        ]
        for s in suggestions:
            if st.button(s, key=f"sug_{s}", use_container_width=True):
                if "messages" not in st.session_state:
                    st.session_state["messages"] = []
                st.session_state["pending_question"] = s
                st.rerun()


# ════════════════════════════════════════
# RIGHT: Chat interface
# ════════════════════════════════════════
with col_right:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;
                color:#e8e6e0;margin-bottom:0.8rem;">
        💬 Ask your document
    </div>
    """, unsafe_allow_html=True)

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Chat container
    chat_container = st.container()

    with chat_container:
        if not st.session_state["messages"]:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;
                        font-family:'DM Mono',monospace;font-size:0.78rem;color:#2a2a4a;">
                <div style="font-size:1.8rem;margin-bottom:0.8rem;">💬</div>
                Upload and index a PDF, then ask anything about it.
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state["messages"]:
                if msg["role"] == "user":
                    st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    sources_html = ""
                    if msg.get("sources"):
                        chips = "".join([
                            f'<span class="source-chip">📄 {s["pdf"]} p.{s["page"]} · {s["score"]}</span>'
                            for s in msg["sources"]
                        ])
                        sources_html = f'<div class="msg-source">Sources: {chips}</div>'
                    st.markdown(
                        f'<div class="msg-assistant">{msg["content"]}{sources_html}</div>',
                        unsafe_allow_html=True
                    )

    # Handle pending question from suggestion buttons
    if "pending_question" in st.session_state:
        pending = st.session_state.pop("pending_question")
        st.session_state["messages"].append({"role": "user", "content": pending})

        if not st.session_state.get("api_key"):
            st.session_state["messages"].append({
                "role": "assistant",
                "content": "⚠️ Please enter your Anthropic API key in the sidebar.",
                "sources": [],
            })
        elif "indexed_docs" not in st.session_state or not st.session_state["indexed_docs"]:
            st.session_state["messages"].append({
                "role": "assistant",
                "content": "⚠️ Please upload and index a PDF first.",
                "sources": [],
            })
        else:
            with st.spinner("Searching document and generating answer..."):
                try:
                    pipeline = load_pipeline()
                    result = get_answer(pending, pipeline)
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                except Exception as e:
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": f"Error: {e}",
                        "sources": [],
                    })
        st.rerun()

    # Input box
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_send = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Question",
                placeholder="Ask anything about your document...",
                label_visibility="collapsed",
            )
        with col_send:
            send = st.form_submit_button("Send", use_container_width=True)

    if send and user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})

        if not st.session_state.get("api_key"):
            st.session_state["messages"].append({
                "role": "assistant",
                "content": "⚠️ Please enter your Anthropic API key in the sidebar.",
                "sources": [],
            })
        elif "indexed_docs" not in st.session_state or not st.session_state["indexed_docs"]:
            st.session_state["messages"].append({
                "role": "assistant",
                "content": "⚠️ Please upload and index a PDF first.",
                "sources": [],
            })
        else:
            with st.spinner("Searching document and generating answer..."):
                try:
                    pipeline = load_pipeline()
                    result = get_answer(user_input, pipeline)
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                except Exception as e:
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": f"Error: {e}",
                        "sources": [],
                    })
        st.rerun()
