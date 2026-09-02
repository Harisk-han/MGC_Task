"""
Part 2 — SQL Database & Analytical Queries Page.
Interactive SQL runner, Schema Inspector, and Analytics.
"""

import sys
from pathlib import Path
import sqlite3
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Part 2: Database & SQL", page_icon="🗄️", layout="wide")

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
    background: linear-gradient(90deg, #34d399, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# Page Header
st.markdown("""
<div class="page-header">
    <div class="page-title">🗄️ Part 2 — SQL Database & Analytical Queries</div>
    <div style="color:#9ca3af;margin-top:0.3rem;">Relational Database Schema, Upstream Deduplication Constraints, and Analytical Queries on SQLite.</div>
</div>
""", unsafe_allow_html=True)

# Load database into in-memory SQLite
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    csv_path = Path(__file__).parent.parent / "data" / "leads.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.to_sql("leads", conn, if_exists="replace", index=False)
    return conn

conn = get_db_connection()

tab1, tab2, tab3 = st.tabs(["📊 Analytical Queries", "📐 Database Schema (schema.sql)", "🔎 Explore Raw Data"])

# Tab 1: Analytical Queries
with tab1:
    st.markdown("### 1️⃣ Query 1: Conversion Rate by Lead Source (>= 200 leads)")
    st.write("Calculates conversion percentage grouped by lead source, filtering for sources with at least 200 total leads.")
    
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
    
    with st.expander("📄 View SQL Query 1 Code", expanded=False):
        st.code(q1_sql, language="sql")
        
    df_q1 = pd.read_sql_query(q1_sql, conn)
    col_chart, col_tbl = st.columns([3, 2])
    with col_chart:
        st.bar_chart(df_q1.set_index("source")["conversion_rate_pct"], color="#60a5fa")
    with col_tbl:
        st.dataframe(df_q1, use_container_width=True)

    st.markdown("---")

    st.markdown("### 2️⃣ Query 2: Duplicate Lead Fingerprint Detector (`crm_record_hash`)")
    st.write("Identifies leads entered multiple times by different sales agents by matching `crm_record_hash` fingerprints.")
    
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
    
    with st.expander("📄 View SQL Query 2 Code", expanded=False):
        st.code(q2_sql, language="sql")
        
    df_q2 = pd.read_sql_query(q2_sql, conn)
    st.metric("Total Duplicate Fingerprints Found", len(df_q2))
    st.dataframe(df_q2.head(20), use_container_width=True)
    st.info("💡 **Schema Level Fix**: Enforcing `crm_record_hash UNIQUE` in `schema.sql` rejects redundant inserts upstream.")

# Tab 2: Schema Inspector
with tab2:
    st.markdown("### 📐 Database Schema Definition (`part2_db/schema.sql`)")
    st.write("The schema normalizes lookup categories (`lead_sources`, `property_types`) and enforces database constraints.")
    
    schema_path = Path(__file__).parent.parent / "part2_db" / "schema.sql"
    if schema_path.exists():
        st.code(schema_path.read_text(encoding="utf-8"), language="sql")

# Tab 3: Explore Raw Data
with tab3:
    st.markdown("### 🔎 Search & Filter `leads.csv`")
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
