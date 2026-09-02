"""
MGC Developments — Single Master Navigation Application.
Enables interactive testing for all project tasks:
  - Overview / Dashboard
  - Part 1: Document Assistant (RAG Q&A)
  - Part 2: SQL Database & Analytical Queries
  - Part 3: ML Lead Conversion Scoring Model
"""

import sys
from pathlib import Path
import sqlite3
import pandas as pd
import joblib
import streamlit as st
from langchain.schema import SystemMessage, HumanMessage

# Ensure sub-modules are importable
sys.path.insert(0, str(Path(__file__).parent / "part1_rag"))
sys.path.insert(0, str(Path(__file__).parent / "part3_ml"))

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
    from train_model import normalize_city
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
    from part3_ml.train_model import normalize_city

# ---------------------------------------------------------------------------
# Page Config & Global Styles
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MGC Sales Intelligence Hub",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1f2937 100%);
    color: #f3f4f6;
}

.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
}

.main-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}

.main-sub {
    color: #9ca3af;
    font-size: 1.05rem;
    margin-bottom: 1.2rem;
}

.stat-badge {
    display: inline-block;
    background: rgba(96, 165, 250, 0.12);
    border: 1px solid rgba(96, 165, 250, 0.3);
    color: #93c5fd;
    border-radius: 9999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.5rem;
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

.badge-high {
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.4);
    color: #4ade80;
    padding: 0.4rem 1rem;
    border-radius: 8px;
    font-weight: 700;
}
.badge-med {
    background: rgba(234, 179, 8, 0.15);
    border: 1px solid rgba(234, 179, 8, 0.4);
    color: #facc15;
    padding: 0.4rem 1rem;
    border-radius: 8px;
    font-weight: 700;
}
.badge-low {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #f87171;
    padding: 0.4rem 1rem;
    border-radius: 8px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# In-Memory SQLite Helper for Part 2
# ---------------------------------------------------------------------------
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    csv_path = Path(__file__).parent / "data" / "leads.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.to_sql("leads", conn, if_exists="replace", index=False)
    return conn

# ---------------------------------------------------------------------------
# Sidebar Navigation Menu
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric-line/100/34d399/company.png", width=60)
    st.markdown("## 🏢 MGC Sales Desk")
    st.markdown("<span style='color:#9ca3af;font-size:0.85rem;'>Master Multi-Task Testing Hub</span>", unsafe_allow_html=True)
    st.markdown("---")

    nav_choice = st.radio(
        "Navigation Menu:",
        [
            "🏠 Overview Dashboard",
            "💬 Part 1: Document Assistant (RAG)",
            "🗄️ Part 2: SQL Database & Analytics",
            "🎯 Part 3: Lead Scoring Model (ML)",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ System Engine")
    st.markdown("• **LLM**: Gemini 2.5 Flash\n• **Embeddings**: models/gemini-embedding-001\n• **Store**: FAISS Index\n• **DB**: SQLite (`leads` table)\n• **ML Model**: Logistic Regression")

# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.markdown("""
<div class="glass-card">
    <div class="main-title">🏢 MGC Developments Intelligence Hub</div>
    <div class="main-sub">Unified Master Testing Suite for Document RAG Q&A, SQL Relational Analytics, and Machine Learning Lead Scoring.</div>
    <div>
        <span class="stat-badge">📄 3 Source Docs</span>
        <span class="stat-badge">🔍 FAISS Vector Store</span>
        <span class="stat-badge">🗄️ SQLite Database</span>
        <span class="stat-badge">📈 PR-AUC 0.3648</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ===========================================================================
# VIEW 1: Overview Dashboard
# ===========================================================================
if nav_choice == "🏠 Overview Dashboard":
    st.markdown("### ⚡ System Overview & Quick Stats")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Grounding Docs", "3 Files", "Brochure, Price List, FAQ")
    with m2:
        st.metric("Vector Index", "FAISS", "Top-5 Chunks")
    with m3:
        st.metric("Cleaned Leads", "9,000 Rows", "160 Duplicates Dropped")
    with m4:
        st.metric("ML PR-AUC", "0.3648", "5.25x Lift over Baseline")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 💬 Part 1: Grounded Document Assistant")
        st.write("Answers salesperson queries strictly using MGC documents. Enforces conflict detection (e.g. 2% vs 2.5% transfer fee), refusal on unpublished metrics (rental yield), and unconfirmed status flags.")
        st.info("👈 Select **💬 Part 1: Document Assistant** from the sidebar to test.")

    with col2:
        st.markdown("#### 🗄️ Part 2: SQL Schema & Analytics")
        st.write("Relational schema with `UNIQUE (crm_record_hash)` constraint for duplicate prevention. Executes live queries for conversion rates by source and duplicate fingerprint detection.")
        st.info("👈 Select **🗄️ Part 2: SQL Database** from the sidebar to test.")

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 🎯 Part 3: ML Lead Conversion Model")
        st.write("Scikit-Learn Logistic Regression pipeline evaluating conversion likelihood. Features deduplication, leakage removal (`token_amount`), city casing normalization, and imbalanced weighting.")
        st.info("👈 Select **🎯 Part 3: Lead Scoring** from the sidebar to test.")

    with col4:
        st.markdown("#### 🌐 Part 4: Unified Master Application")
        st.write("This single multi-navigation interface combines all project tasks into one seamless testing environment.")

# ===========================================================================
# VIEW 2: Part 1 — Document Assistant (RAG)
# ===========================================================================
elif nav_choice == "💬 Part 1: Document Assistant (RAG)":
    st.markdown("### 💬 Part 1: Document Assistant (RAG)")
    st.write("Ask any question about the project brochure, price list, floor plans, or booking policies. Answers are strictly grounded in official MGC documents.")

    st.markdown("#### 💡 Quick Test Questions:")
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

    if "master_rag_messages" not in st.session_state:
        st.session_state.master_rag_messages = []

    for msg in st.session_state.master_rag_messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("docs"):
                with st.expander("📚 View Grounding Context Chunks"):
                    for doc in msg["docs"]:
                        src = doc.metadata.get("source", "?")
                        sec = doc.metadata.get("section", "?")
                        st.markdown(f'<span class="source-tag">📄 {src} › {sec}</span>', unsafe_allow_html=True)
                        st.caption(doc.page_content)

    user_input = st.chat_input("Ask about MGC Aurora Heights...")
    active_prompt = suggested_q or user_input

    if active_prompt:
        st.session_state.master_rag_messages.append({"role": "user", "content": active_prompt})
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

                    with st.expander("📚 View Grounding Context Chunks"):
                        for doc in relevant_docs:
                            src = doc.metadata.get("source", "?")
                            sec = doc.metadata.get("section", "?")
                            st.markdown(f'<span class="source-tag">📄 {src} › {sec}</span>', unsafe_allow_html=True)
                            st.caption(doc.page_content)

                    st.session_state.master_rag_messages.append({
                        "role": "assistant",
                        "content": answer,
                        "docs": relevant_docs
                    })
                except Exception as e:
                    st.error(f"Error querying assistant: {e}\n\nEnsure GOOGLE_API_KEY is set in `.env`.")

# ===========================================================================
# VIEW 3: Part 2 — SQL Database & Analytics
# ===========================================================================
elif nav_choice == "🗄️ Part 2: SQL Database & Analytics":
    st.markdown("### 🗄️ Part 2: SQL Database & Analytical Queries")
    st.write("Relational database schema, constraint definition, and live query execution.")

    conn = get_db_connection()
    tab1, tab2, tab3 = st.tabs(["📊 Analytical Queries", "📐 Database Schema (schema.sql)", "🔎 Explore Raw Data"])

    with tab1:
        st.markdown("#### 1️⃣ Query 1: Conversion Rate by Lead Source (>= 200 leads)")
        q1_sql = """
        SELECT
            source,
            COUNT(*) AS total_leads,
            SUM(CASE WHEN converted = 1 THEN 1 ELSE 0 END) AS converted_leads,
            ROUND(100.0 * SUM(CASE WHEN converted = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS conversion_rate_pct
        FROM leads
        GROUP BY source
        HAVING COUNT(*) >= 200
        ORDER BY conversion_rate_pct DESC;
        """
        with st.expander("📄 View SQL Query 1 Code"):
            st.code(q1_sql, language="sql")
        df_q1 = pd.read_sql_query(q1_sql, conn)
        c_chart, c_tbl = st.columns([3, 2])
        with c_chart:
            st.bar_chart(df_q1.set_index("source")["conversion_rate_pct"], color="#60a5fa")
        with c_tbl:
            st.dataframe(df_q1, use_container_width=True)

        st.markdown("---")

        st.markdown("#### 2️⃣ Query 2: Duplicate Lead Fingerprint Detector (`crm_record_hash`)")
        q2_sql = """
        SELECT
            crm_record_hash,
            COUNT(*) AS entry_count,
            GROUP_CONCAT(lead_id, ', ') AS lead_ids
        FROM leads
        GROUP BY crm_record_hash
        HAVING COUNT(*) > 1
        ORDER BY entry_count DESC;
        """
        with st.expander("📄 View SQL Query 2 Code"):
            st.code(q2_sql, language="sql")
        df_q2 = pd.read_sql_query(q2_sql, conn)
        st.metric("Total Duplicate Fingerprints Found", len(df_q2))
        st.dataframe(df_q2.head(20), use_container_width=True)

    with tab2:
        st.markdown("#### 📐 Database Schema (`part2_db/schema.sql`)")
        schema_path = Path(__file__).parent / "part2_db" / "schema.sql"
        if schema_path.exists():
            st.code(schema_path.read_text(encoding="utf-8"), language="sql")

    with tab3:
        st.markdown("#### 🔎 Search & Filter `leads.csv` Data")
        df_all = pd.read_sql_query("SELECT * FROM leads LIMIT 1000", conn)
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            selected_source = st.multiselect("Filter Source", df_all["source"].unique())
        with col_f2:
            converted_filter = st.selectbox("Filter Converted Status", ["All", "Converted Only (1)", "Not Converted (0)"])
        
        df_filtered = df_all.copy()
        if selected_source:
            df_filtered = df_filtered[df_filtered["source"].isin(selected_source)]
        if converted_filter == "Converted Only (1)":
            df_filtered = df_filtered[df_filtered["converted"] == 1]
        elif converted_filter == "Not Converted (0)":
            df_filtered = df_filtered[df_filtered["converted"] == 0]
        st.dataframe(df_filtered, use_container_width=True)

# ===========================================================================
# VIEW 4: Part 3 — ML Lead Conversion Model
# ===========================================================================
elif nav_choice == "🎯 Part 3: Lead Scoring Model (ML)":
    st.markdown("### 🎯 Part 3: ML Lead Conversion Model")
    st.write("Predictive Machine Learning Pipeline evaluating conversion likelihood on imbalanced lead data (PR-AUC = 0.3648).")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Primary Metric (PR-AUC)", "0.3648", delta="5.25x Lift")
    with m2:
        st.metric("Random Baseline", "0.0694")
    with m3:
        st.metric("ROC-AUC Score", "0.8384")
    with m4:
        st.metric("Class Ratio", "93% / 7%")

    st.markdown("---")

    MODEL_PATH = Path(__file__).parent / "part3_ml" / "model.joblib"
    if not MODEL_PATH.exists():
        st.error("⚠️ Trained model `model.joblib` not found. Run `python part3_ml/train_model.py` first.")
    else:
        model = joblib.load(MODEL_PATH)

        with st.form("master_ml_scoring_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**1️⃣ Source & Property**")
                source = st.selectbox("Lead Source", [
                    "Facebook Ads", "Property Portal", "Google Search", "Instagram",
                    "Referral", "Walk-in", "WhatsApp Campaign", "Expo Stall", "Billboard",
                ])
                city = st.text_input("City", "Islamabad")
                area = st.text_input("Area", "Bahria Town")
                property_type = st.selectbox("Property Type", [
                    "Apartment", "Plot", "Villa", "Commercial Shop", "Penthouse", "Farmhouse",
                ])

            with col2:
                st.markdown("**2️⃣ Demographics & Financials**")
                budget_pkr_lac = st.number_input("Budget (PKR lac)", min_value=0.0, value=250.0, step=10.0)
                bedrooms = st.number_input("Bedrooms (0 for Shop/Plot)", min_value=0, max_value=10, value=2)
                agent_experience_years = st.number_input("Agent Experience (Years)", min_value=0.0, max_value=30.0, value=3.0)
                created_dow = st.selectbox(
                    "Day Created",
                    options=list(range(7)),
                    format_func=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d],
                )

            with col3:
                st.markdown("**3️⃣ Engagement & Activities**")
                first_response_minutes = st.number_input("First Response Time (min)", min_value=0, value=20)
                calls_made = st.number_input("Calls Made", min_value=0, value=3)
                total_call_seconds = st.number_input("Total Call Time (sec)", min_value=0, value=120)
                whatsapp_replies = st.number_input("WhatsApp Replies", min_value=0, value=2)
                site_visits = st.number_input("Site Visits Completed", min_value=0, value=1)

            st.markdown("---")
            st.markdown("**Buyer Attributes**")
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                is_overseas = st.checkbox("✈️ Overseas Buyer")
            with col_b2:
                referred_by_existing_client = st.checkbox("⭐ Referred by Existing Client")
            with col_b3:
                has_financing_approved = st.checkbox("🏦 Pre-Approved Financing")

            submit_btn = st.form_submit_button("🔥 Compute Booking Probability Score", use_container_width=True)

        if submit_btn:
            row = pd.DataFrame([{
                "source": source,
                "city": normalize_city(city),
                "area": area,
                "property_type": property_type,
                "budget_pkr_lac": budget_pkr_lac,
                "bedrooms": bedrooms,
                "first_response_minutes": first_response_minutes,
                "calls_made": calls_made,
                "total_call_seconds": total_call_seconds,
                "whatsapp_replies": whatsapp_replies,
                "site_visits": site_visits,
                "agent_experience_years": agent_experience_years,
                "is_overseas": int(is_overseas),
                "referred_by_existing_client": int(referred_by_existing_client),
                "has_financing_approved": int(has_financing_approved),
                "created_dow": created_dow,
            }])

            proba = model.predict_proba(row)[0, 1]
            pct = proba * 100

            st.markdown("### 📈 Scoring Analysis & Recommendation")
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.metric("Conversion Likelihood", f"{pct:.1f}%")
                if pct >= 40:
                    st.markdown('<span class="badge-high">🔥 HIGH POTENTIAL</span>', unsafe_allow_html=True)
                elif pct >= 15:
                    st.markdown('<span class="badge-med">⚡ MODERATE POTENTIAL</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-low">❄️ LOW POTENTIAL</span>', unsafe_allow_html=True)

            with res_col2:
                st.progress(min(int(pct), 100))
                if pct >= 40:
                    st.success("🎯 **Priority Follow-up**: Lead shows strong purchase signals. Senior sales agent recommended.")
                elif pct >= 15:
                    st.warning("💬 **Active Engagement**: Send floor plans and financing options via WhatsApp.")
                else:
                    st.info("📧 **Nurture Drip**: Include lead in automated promotional newsletter drip.")
