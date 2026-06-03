import os
import traceback as tb
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from ingest import ingest_pdf
from rag_chain import build_rag_chain, ask

load_dotenv()

# ── Shared state ─────────────────────────────────────────────────────────────
_chain_tuple = None
_startup_error = None
PDF_PATH = os.getenv("PDF_PATH", r"C:\Users\Aryan\Documents\med\Medical_book.pdf")
FRONTEND_DIR = Path(__file__).parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chain_tuple, _startup_error
    print("=" * 60)
    print("⏳  STEP 1/2: Ingesting PDF...")
    print(f"    PDF path: {PDF_PATH}")
    print(f"    File exists: {os.path.exists(PDF_PATH)}")
    print("=" * 60)
    try:
        ingest_pdf(PDF_PATH)
        print("=" * 60)
        print("⏳  STEP 2/2: Building RAG chain (may download model ~900MB)...")
        print("=" * 60)
        _chain_tuple = build_rag_chain()
        print("=" * 60)
        print("✅  RAG chain is READY. Server is fully operational.")
        print("=" * 60)
    except Exception as e:
        _startup_error = tb.format_exc()
        print("=" * 60)
        print("❌  STARTUP FAILED:")
        print(_startup_error)
        print("=" * 60)
    yield
    print("🛑 Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Medical RAG API",
    description="Ask questions about your medical PDF.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[dict]


# ── Routes — MUST be before app.mount() ──────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok" if _chain_tuple else "error",
        "chain_loaded": _chain_tuple is not None,
        "startup_error": _startup_error,
    }


@app.get("/debug")
async def debug():
    """Shows full startup error and env status for debugging."""
    return {
        "chain_loaded": _chain_tuple is not None,
        "startup_error": _startup_error,
        "pdf_path": PDF_PATH,
        "pdf_exists": os.path.exists(PDF_PATH),
        "pinecone_index": os.getenv("PINECONE_INDEX_NAME"),
        "hf_token_set": bool(os.getenv("HF_TOKEN")),
        "huggingface_token_set": bool(os.getenv("HUGGINGFACEHUB_API_TOKEN")),
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(body: QuestionRequest):
    if _chain_tuple is None:
        detail = f"RAG chain not ready. Startup error: {_startup_error}" if _startup_error else "RAG chain is still loading..."
        raise HTTPException(status_code=503, detail=detail)
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = ask(_chain_tuple, body.question)
        return AnswerResponse(
            answer=result["answer"],
            sources=result["sources"],
        )
    except Exception as e:
        error_detail = tb.format_exc()
        print("\n" + "=" * 60)
        print("❌ /ask ERROR:")
        print(error_detail)
        print("=" * 60 + "\n")
        # Return full traceback in detail so you can see it in browser
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/test-ask")
async def test_ask(q: str = "What is diabetes?"):
    """Test the chain directly — open in browser to see exact error."""
    if _chain_tuple is None:
        return {"error": "chain not loaded", "startup_error": _startup_error}
    try:
        result = ask(_chain_tuple, q)
        return {"ok": True, "answer": result["answer"], "sources_count": len(result["sources"])}
    except Exception as e:
        return {"ok": False, "error": str(e), "traceback": tb.format_exc()}


@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ── Static files — MUST come last ────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
