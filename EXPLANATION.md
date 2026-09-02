# MGC Build Task — Complete Explanation Guide

Welcome to the **MGC Build Task** repository documentation. This guide breaks down the codebase architecture, design decisions, database schemas, machine learning pipeline, and instructions on how to run every part of the system.

---

## 📁 Repository Overview

```
mgc-build/
├── docs/                # Raw source documents (brochure, price list, booking policy)
├── data/                # Raw dataset (leads.csv with 9,160 lead records)
├── part1_rag/           # Part 1: Document Q&A RAG Assistant (rag.py, app.py, faiss_index/)
├── part2_db/            # Part 2: SQL Database Schema & Queries (schema.sql, queries.sql)
├── part3_ml/            # Part 3: ML Lead Conversion Model (train_model.py, model.joblib)
├── part4_web/           # Part 4: Streamlit Web UI combining Parts 1 & 3 (app.py)
├── .env                 # Environment variables (GOOGLE_API_KEY)
├── EXPLANATION.md        # Complete technical explanation guide
├── README.md            # Brief project overview & instructions
└── requirements.txt     # Python dependencies
```

---

## 🧠 Part 1 — Document Q&A Assistant (RAG)

### Goal
Build a grounded Q&A assistant for MGC salespeople that answers questions about project pricing, floor plans, booking policies, and project details **strictly using official MGC documents**.

### Architecture
The implementation follows a **Retrieve → Augment → Generate (RAG)** pipeline:
1. **Document Loading & Sectioning**: Markdown documents in `docs/` (`01_project_brochure.md`, `02_price_list_payment_plan.md`, `03_booking_policy_faq.md`) are parsed and split by level-2 markdown headings (`## `) to preserve section metadata.
2. **Chunking**: Using LangChain's `RecursiveCharacterTextSplitter` (chunk size: 500 characters, overlap: 80 characters).
3. **Vector Store & Embeddings**: Chunks are embedded using Google's `models/gemini-embedding-001` and indexed in a local **FAISS** vector store saved to `part1_rag/faiss_index/`.
4. **Retrieval**: On any question, top 5 (`TOP_K = 5`) most relevant text chunks are retrieved via cosine similarity.
5. **Grounded Generation**: The retrieved context and question are sent to **Google Gemini 2.5 Flash** with a strict system prompt enforcing strict factual discipline.

### Prompting & Hard Case Discipline
- **Conflict Flagging (Transfer Fee)**: The price list states a 2% transfer fee before possession, while the booking policy FAQ states 2.5%. The model is instructed **never to pick one number**, but rather to state both figures, cite both documents, and flag the discrepancy for double-checking.
- **Refusal on Unpublished Information (Rental Yield)**: The FAQ explicitly states MGC does not guarantee rental yield. When asked for rental yields, the model refuses to estimate and directs the user to contact the marketing manager.
- **Unconfirmed Status (Anchor Tenant)**: The brochure notes anchor tenants are unconfirmed; the model accurately states this status without implying a contract has been signed.
- **Price Calculations**: Computes base price plus stacked corner/park-view premiums dynamically.

---

## 🗄️ Part 2 — SQL Database & Analytical Queries

### 1. Database Schema (`part2_db/schema.sql`)
- Includes normalized lookup tables for controlled fields (`lead_sources`, `property_types`).
- Core table `leads` enforces data integrity and foreign key constraints.
- **Upstream Duplicate Prevention**: Enforces a `UNIQUE` constraint on `crm_record_hash`. When multiple sales agents attempt to enter the same lead (identical hash), the second insert is rejected at the database level rather than creating duplicate leads.

### 2. Analytical Queries (`part2_db/queries.sql`)
- **Query 1 — Conversion Rate by Source**: Filters lead sources with at least 200 total leads, calculates conversion rate percentage, and orders results by highest conversion rate first.
- **Query 2 — Duplicate Lead Detection**: Groups leads by `crm_record_hash` having `COUNT(*) > 1` and aggregates all duplicate `lead_id` strings (using `STRING_AGG` / `GROUP_CONCAT`).

---

## 📊 Part 3 — Machine Learning Lead Conversion Model

### Goal
Train a predictive model to estimate the probability (`0.0` to `1.0`) that a sales lead converts into a confirmed booking, allowing sales teams to prioritize outreach.

### Key Data Decisions & Cleaning
1. **Deduplication First**: ~160 duplicate leads (matching `crm_record_hash`) were identified. Only the first occurrence was kept to prevent data leakage between train and test splits.
2. **Leakage Dropped (`token_amount_received_pkr`)**: 100% of converted leads had a non-zero token amount, while non-converted leads had 0. Because token amounts are recorded *after* booking, keeping this feature would create artificial 100% accuracy in training but fail completely on new un-converted incoming leads.
3. **Identifiers Dropped**: `lead_id` and `crm_record_hash` were removed as non-predictive identifiers.
4. **City Casing Normalization**: Messy raw entries (`ISLAMABAD`, `isb`, `Rwp`, `khi`) were mapped to canonical city names (`Islamabad`, `Rawalpindi`, `Karachi`) to prevent sparse, split category categories in one-hot encoding.
5. **Bedrooms Logic**: Missing bedroom values (~40%) occur when property types are Commercial Shops or Plots. These were filled with `0` (not applicable) rather than the mean.
6. **Imputation**: Missing values for `budget_pkr_lac` and `agent_experience_years` were imputed using the median. Missing `area` was encoded as an explicit `"Unknown"` category.

### Metric Selection: PR-AUC (Average Precision)
The target class is heavily imbalanced (**~93% Not Converted / ~7% Converted**). 
- Standard **Accuracy** is misleading (a naive model predicting "never converts" achieves ~93% accuracy but is useless).
- **ROC-AUC** can be artificially inflated by large true negative counts.
- **PR-AUC (Precision-Recall Area Under Curve)** was chosen as the primary metric because it directly evaluates how effectively the model ranks rare positive converted leads.

### Performance Results
- **PR-AUC**: **0.3648** (vs **0.0694** random baseline — over **5x lift**)
- **ROC-AUC**: **0.8384**
- **Model Saved**: Serialized pipeline stored at `part3_ml/model.joblib`.

---

## 🌐 Part 4 — Unified Streamlit Web Application

The Streamlit web UI (`part4_web/app.py`) provides an interactive interface with two main tabs:

1. **Tab 1: Ask the Document Assistant**:
   - Interactive Q&A chat interface powered by `part1_rag/rag.py`.
   - Allows sales agents to ask questions about project documents in real-time.

2. **Tab 2: Score a Lead**:
   - Web form to input lead parameters (Source, City, Budget, Property Type, Response Time, Calls Made, Site Visits, Agent Experience, etc.).
   - Loads `part3_ml/model.joblib` to calculate and display the live conversion probability score.

---

## 🛠️ How to Run

### Setup Environment
1. Ensure dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your Google Gemini API Key in `.env`:
   ```env
   GOOGLE_API_KEY=AIzaSy...
   ```

### Command Execution

- **Test Part 1 (CLI RAG Q&A)**:
  ```bash
  python part1_rag/rag.py "What's the transfer fee?"
  ```

- **Train Part 3 (ML Model)**:
  ```bash
  python part3_ml/train_model.py
  ```

- **Run Part 4 (Unified Streamlit Web UI)**:
  ```bash
  streamlit run part4_web/app.py
  ```

- **Run Part 1 Standalone Chat UI**:
  ```bash
  streamlit run part1_rag/app.py
  ```
