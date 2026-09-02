"""
MGC document assistant — scalable grounded Q&A using a FAISS vector store.

Architecture (Retrieve → Augment → Generate):
  1. Load & chunk all .md docs from /docs using LangChain text splitters,
     preserving source filename and section heading as metadata.
  2. Embed every chunk with Google's Gemini embedding model and index them
     in a FAISS vector store persisted to part1_rag/faiss_index/.
  3. On each query, embed the question, retrieve the top-k most relevant
     chunks via cosine similarity, and pass ONLY those chunks to Gemini
     as grounding context — not the entire corpus.
  4. A strict system prompt enforces citation, conflict flagging, and refusal
     on missing information.

Why FAISS?
  • Zero server / infra setup — runs in-process.
  • Index persists to disk: embeddings are computed once, reused on every
    subsequent run (much faster).
  • Production-swap path: swap FAISS for Pinecone / Weaviate / Chroma with
    one line change in get_vectorstore().

LLM  : Google Gemini 2.5 Flash  (langchain-google-genai)
Embed: Google models/embedding-001
Store: FAISS (langchain-community)

Set GOOGLE_API_KEY in .env before running.
Usage:
    python part1_rag/rag.py "What's the transfer fee?"
    python part1_rag/rag.py --rebuild "What's the transfer fee?"   # force re-index
"""

import asyncio
import os
import re
import sys
from pathlib import Path

# Ensure UTF-8 stdout/stderr encoding for Windows terminals with emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def _ensure_event_loop():
    """Ensure an asyncio event loop exists in the current thread (required for gRPC in Streamlit)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document, SystemMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

# Load .env from project root (one level above this file's directory)
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOCS_DIR        = Path(__file__).parent.parent / "docs"
FAISS_INDEX_DIR = Path(__file__).parent / "faiss_index"

GEMINI_MODEL     = "gemini-2.5-flash"
EMBEDDING_MODEL  = "models/gemini-embedding-001"

# Number of chunks to retrieve per query — increase for broader coverage,
# decrease to keep context tight. At scale (100s of docs), 5–8 is typical.
TOP_K = 5

# Chunk size in characters; overlap keeps context at boundaries intact.
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 80

SYSTEM_PROMPT = """\
You are a sales-desk assistant for MGC Developments. You answer questions from \
MGC salespeople using ONLY the document excerpts provided in the context below. \
Follow these rules exactly:

1. Answer only from the provided context. Never use outside knowledge about real \
estate, pricing, or policy — even if you think you know the answer.
2. Always cite the source document (and section, if relevant) for every fact you state.
3. If two sources give different figures for the same thing, DO NOT pick one. \
State both values, both sources, and flag the conflict explicitly so the \
salesperson knows to double-check before quoting a customer.
4. If the answer is not in the context, say so plainly. Do not estimate, infer, or \
invent a number. Point the salesperson to who they should ask instead (e.g. the \
marketing manager), if the context says who that is.
5. If a document explicitly says something is unconfirmed / not yet decided, say \
that it is unconfirmed — do not imply a decision has been made.
6. Do calculations (e.g. stacking premiums onto a base price) yourself when the \
context gives the components, and show the arithmetic briefly.

Format:
Answer: <your answer>
Source(s): <document name(s) + section>
"""

# ---------------------------------------------------------------------------
# Document loading & chunking
# ---------------------------------------------------------------------------

def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """
    Load every .md file from docs_dir as LangChain Documents.
    Each document carries metadata: source filename and detected section heading.
    """
    docs: list[Document] = []
    for path in sorted(docs_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        # Split on level-2 headings so each section gets its own metadata tag
        sections = re.split(r"\n(?=## )", content)
        for section in sections:
            section = section.strip()
            if not section:
                continue
            heading_match = re.match(r"##\s*(.+)", section)
            heading = heading_match.group(1).strip() if heading_match else "Intro"
            docs.append(Document(
                page_content=section,
                metadata={"source": path.name, "section": heading},
            ))
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split documents into smaller overlapping chunks for embedding.
    Metadata (source, section) is preserved on every child chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# Embeddings & vector store
# ---------------------------------------------------------------------------

def build_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return a Gemini embedding model. Reads GOOGLE_API_KEY from env."""
    _ensure_event_loop()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Get a key at https://aistudio.google.com/app/apikey"
        )
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )


def build_vectorstore(
    chunks: list[Document],
    embeddings: GoogleGenerativeAIEmbeddings,
) -> FAISS:
    """
    Embed all chunks and save the FAISS index to FAISS_INDEX_DIR.
    Re-run whenever source documents change (or use --rebuild flag).
    """
    print(f"  Building FAISS index from {len(chunks)} chunks...", flush=True)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_DIR))
    print(f"  Index saved to {FAISS_INDEX_DIR}", flush=True)
    return vectorstore


def load_vectorstore(embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
    """Load a previously-saved FAISS index from disk."""
    return FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,   # safe: we wrote this index ourselves
    )


def get_vectorstore(
    embeddings: GoogleGenerativeAIEmbeddings,
    rebuild: bool = False,
) -> FAISS:
    """
    Smart loader:
      - If rebuild=True or no index on disk → build from scratch and save.
      - Otherwise           → load from disk (fast path).
    """
    index_exists = (FAISS_INDEX_DIR / "index.faiss").exists()
    if rebuild or not index_exists:
        docs   = load_documents()
        chunks = chunk_documents(docs)
        return build_vectorstore(chunks, embeddings)
    else:
        print(f"  Loading FAISS index from {FAISS_INDEX_DIR}", flush=True)
        return load_vectorstore(embeddings)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def build_llm() -> ChatGoogleGenerativeAI:
    """Construct the Gemini LLM via LangChain."""
    _ensure_event_loop()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set.")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.0,
        max_output_tokens=600,
    )


# ---------------------------------------------------------------------------
# RAG answer function
# ---------------------------------------------------------------------------

def format_retrieved_context(docs: list[Document]) -> str:
    """
    Format retrieved chunks into a labelled context block identical in style
    to the old 'stuff all docs' format, so the system prompt rules still apply.
    """
    parts = []
    for doc in docs:
        source  = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section", "—")
        parts.append(
            f"--- SOURCE: {source} | SECTION: {section} ---\n{doc.page_content}"
        )
    return "\n\n".join(parts)


def answer_question(
    question: str,
    llm: ChatGoogleGenerativeAI | None = None,
    embeddings: GoogleGenerativeAIEmbeddings | None = None,
    rebuild: bool = False,
) -> str:
    """
    Answer *question* using vector-retrieved MGC doc chunks as grounding context.

    Args:
        question:   The salesperson's question.
        llm:        Optional pre-built LangChain LLM (created if None).
        embeddings: Optional pre-built embedding model (created if None).
        rebuild:    If True, force FAISS index rebuild from source docs.

    Returns:
        Grounded answer string with citations.
    """
    _ensure_event_loop()
    if llm is None:
        llm = build_llm()
    if embeddings is None:
        embeddings = build_embeddings()

    # Retrieve the most relevant chunks for this question
    vectorstore = get_vectorstore(embeddings, rebuild=rebuild)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    relevant_docs = retriever.invoke(question)

    context = format_retrieved_context(relevant_docs)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {question}"),
    ]

    response = llm.invoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# CLI entry-point — interactive chat loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args    = sys.argv[1:]
    rebuild = "--rebuild" in args
    if rebuild:
        args.remove("--rebuild")

    # Pre-build shared LLM and embeddings once so every query is fast
    print("🔧 Initialising MGC Sales Assistant...", flush=True)
    _llm        = build_llm()
    _embeddings = build_embeddings()
    _vs         = get_vectorstore(_embeddings, rebuild=rebuild)   # load/build index once

    # Monkey-patch answer_question to reuse the already-loaded vectorstore
    def _answer(question: str) -> str:
        retriever     = _vs.as_retriever(search_kwargs={"k": TOP_K})
        relevant_docs = retriever.invoke(question)
        context       = format_retrieved_context(relevant_docs)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {question}"),
        ]
        return _llm.invoke(messages).content

    # If a question was passed as a CLI arg, answer it and exit
    if args:
        print(f"\nQ: {args[0]}\n")
        print(_answer(args[0]))
        sys.exit(0)

    # Otherwise, start the interactive chat loop
    print("\n✅ Ready! Type your question and press Enter. Type 'exit' to quit.\n")
    print("─" * 60)

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        print("\n🤖 MGC Assistant:\n")
        print(_answer(user_input))
        print("\n" + "─" * 60)
