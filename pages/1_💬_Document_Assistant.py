"""
Part 1 — Document Q&A Assistant (RAG) Page.
Grounded Q&A powered by Google Gemini 2.5 Flash + FAISS Vector Store.
"""

import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "part1_rag"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.schema import SystemMessage, HumanMessage

try:
    from rag import (
        build_llm,
        build_embeddings,
        get_vectorstore,
        format_retrieved_context,
        SYSTEM_PROMPT,
        TOP_K,
        answer_question,
    )
except ImportError:
    from part1_rag.rag import (
        build_llm,
        build_embeddings,
        get_vectorstore,
        format_retrieved_context,
        SYSTEM_PROMPT,
        TOP_K,
        answer_question,
    )

st.set_page_config(page_title="Part 1: Document Assistant", page_icon="💬", layout="wide")

# Custom Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.page-header {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.page-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.source-tag {
    background: rgba(167, 139, 250, 0.15);
    border: 1px solid rgba(167, 139, 250, 0.3);
    color: #c4b5fd;
    font-size: 0.75rem;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# Page Header
st.markdown("""
<div class="page-header">
    <div class="page-title">💬 Part 1 — Document Q&A Assistant</div>
    <div style="color:#9ca3af;margin-top:0.3rem;">Grounded Sales Assistant powered by Google Gemini 2.5 Flash + FAISS Vector Store.</div>
</div>
""", unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.markdown("### 📄 Grounding Corpus")
    st.markdown("• `01_project_brochure.md`\n• `02_price_list_payment_plan.md`\n• `03_booking_policy_faq.md` ")
    st.markdown("---")
    st.markdown("### 🎯 Refusal & Conflict Rules")
    st.markdown("1. **Conflict Flagging**: Transfer fee (2% vs 2.5%) stated and flagged explicitly.\n2. **Refusal Discipline**: Rejects rental yield estimations.\n3. **Unconfirmed Status**: States anchor tenant is unconfirmed.")

# Quick Preset Buttons
st.markdown("#### 💡 Quick Test Questions")
q_col1, q_col2, q_col3, q_col4 = st.columns(4)
suggested_q = None

with q_col1:
    if st.button("💸 Transfer Fee Conflict", use_container_width=True):
        suggested_q = "What's the transfer fee?"
with q_col2:
    if st.button("🏢 Available Unit Types", use_container_width=True):
        suggested_q = "What unit types are available?"
with q_col3:
    if st.button("📈 Rental Yield Refusal", use_container_width=True):
        suggested_q = "What is the guaranteed rental yield?"
with q_col4:
    if st.button("🤝 Anchor Tenant Status", use_container_width=True):
        suggested_q = "Who is the anchor tenant?"

st.markdown("---")

# Session Chat State
if "p1_messages" not in st.session_state:
    st.session_state.p1_messages = []

# Render Chat History
for msg in st.session_state.p1_messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg.get("docs"):
            with st.expander("📚 Retrieved Context Chunks"):
                for doc in msg["docs"]:
                    src = doc.metadata.get("source", "?")
                    sec = doc.metadata.get("section", "?")
                    st.markdown(f'<span class="source-tag">📄 {src} › {sec}</span>', unsafe_allow_html=True)
                    st.caption(doc.page_content)

# Chat Input
user_input = st.chat_input("Ask about MGC Aurora Heights...")
active_prompt = suggested_q or user_input

if active_prompt:
    st.session_state.p1_messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(active_prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching FAISS index & invoking Gemini 2.5 Flash..."):
            try:
                embeddings = build_embeddings()
                vs = get_vectorstore(embeddings, rebuild=False)
                retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
                relevant_docs = retriever.invoke(active_prompt)
                
                answer = answer_question(active_prompt)
                st.markdown(answer)
                
                with st.expander("📚 Retrieved Context Chunks"):
                    for doc in relevant_docs:
                        src = doc.metadata.get("source", "?")
                        sec = doc.metadata.get("section", "?")
                        st.markdown(f'<span class="source-tag">📄 {src} › {sec}</span>', unsafe_allow_html=True)
                        st.caption(doc.page_content)

                st.session_state.p1_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "docs": relevant_docs
                })
            except Exception as e:
                st.error(f"Error querying assistant: {e}\n\nEnsure GOOGLE_API_KEY is set in `.env`.")
