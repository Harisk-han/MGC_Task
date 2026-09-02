"""
Part 3 — ML Lead Conversion Model Page.
Interactive Lead Scoring, Feature Engineering Explanation, & Model Metrics.
"""

import sys
from pathlib import Path
import pandas as pd
import joblib
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "part3_ml"))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from train_model import normalize_city
except ImportError:
    from part3_ml.train_model import normalize_city

st.set_page_config(page_title="Part 3: Lead Scoring Model", page_icon="🎯", layout="wide")

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
    background: linear-gradient(90deg, #f472b6, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
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

# Page Header
st.markdown("""
<div class="page-header">
    <div class="page-title">🎯 Part 3 — ML Lead Conversion Model</div>
    <div style="color:#9ca3af;margin-top:0.3rem;">Predictive Machine Learning Pipeline evaluating conversion likelihood on imbalanced lead data (PR-AUC = 0.3648).</div>
</div>
""", unsafe_allow_html=True)

# Model Metrics Banner
st.markdown("### 📊 Model Performance & Baseline")
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

# Main Calculator Form
st.markdown("### 📋 Interactive Lead Scoring Form")

MODEL_PATH = Path(__file__).parent.parent / "part3_ml" / "model.joblib"
if not MODEL_PATH.exists():
    st.error("⚠️ Trained model `model.joblib` not found. Run `python part3_ml/train_model.py` first.")
else:
    model = joblib.load(MODEL_PATH)

    with st.form("ml_scoring_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**1️⃣ Source & Property Attributes**")
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
            st.markdown("**2️⃣ Financials & Agent Profile**")
            budget_pkr_lac = st.number_input("Budget (PKR lac)", min_value=0.0, value=250.0, step=10.0)
            bedrooms = st.number_input("Bedrooms (0 for Shop/Plot)", min_value=0, max_value=10, value=2)
            agent_experience_years = st.number_input("Agent Experience (Years)", min_value=0.0, max_value=30.0, value=3.0)
            created_dow = st.selectbox(
                "Day Created",
                options=list(range(7)),
                format_func=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d],
            )

        with col3:
            st.markdown("**3️⃣ Lead Activity & Engagement**")
            first_response_minutes = st.number_input("First Response Time (min)", min_value=0, value=20)
            calls_made = st.number_input("Calls Made", min_value=0, value=3)
            total_call_seconds = st.number_input("Total Call Time (sec)", min_value=0, value=120)
            whatsapp_replies = st.number_input("WhatsApp Replies", min_value=0, value=2)
            site_visits = st.number_input("Site Visits Completed", min_value=0, value=1)

        st.markdown("---")
        st.markdown("**Buyer Indicators**")
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
                st.success("🎯 **Priority Follow-up**: Lead shows strong purchase signals. Direct call from senior sales agent recommended within 2 hours.")
            elif pct >= 15:
                st.warning("💬 **Active Engagement**: Send tailored floor plans and payment plan details via WhatsApp within 24 hours.")
            else:
                st.info("📧 **Nurture Drip**: Include lead in automated promotional newsletter and campaign updates.")

st.markdown("---")
with st.expander("🔬 View Model Decisions & Feature Engineering Notes"):
    st.markdown("""
    - **Data Leakage Removal**: Dropped `token_amount_received_pkr` because it's only set *after* conversion occurs.
    - **City Normalization**: Maps messy casing (`ISB`, `ISLAMABAD`, `Rwp`) to standard names.
    - **Imbalanced Class Handling**: Uses `class_weight='balanced'` to prevent bias toward the 93% non-converting majority class.
    """)
