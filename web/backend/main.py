import asyncio
import os
import re
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import PyPDF2
import faiss
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import ollama

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RAGAS_DIR = ROOT_DIR / "tests" / "TA" / "ragas"
if str(RAGAS_DIR) not in sys.path:
    sys.path.insert(0, str(RAGAS_DIR))

from minirag import MiniRAG, QueryParam
from minirag.llm.hf import hf_embed
from minirag.llm.ollama import ollama_model_complete
from minirag.prompt_id_revisi import PROMPTS
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

# ragas_*.py (reused as-is below) hard-assert these two env vars at import time
# even though CLEANLAB_TLM_API_KEY is never actually used by the code; stub
# them so importing those modules doesn't force a Cleanlab/OpenAI account
# (mirrors tests/TA/ragas/eval_ragas.py).
os.environ.setdefault("CLEANLAB_TLM_API_KEY", "unused")
os.environ.setdefault("OPENAI_API_KEY", "not-needed-for-ollama-judge")

# Extractor classes reused as-is from the thesis's production RAGAS pipeline
# (tests/TA/ragas/) so the web demo's scores are computed with the exact same
# methodology as the offline experiments -- not a separate ad-hoc reimplementation.
import ragas_faith
import ragas_context_recall
import ragas_ans_correctness
from eval_ragas import (
    extract_faithfulness,
    extract_context_recall,
    extract_answer_correctness,
    truncate_context,
)

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11500")
# Independent from the answering model on principle (kept as a separate named
# constant, matching tests/TA/ragas/eval_ragas.py's convention) even though it
# happens to be the same model here.
RAGAS_EVAL_MODEL = os.getenv("RAGAS_EVAL_MODEL", "gemma3:12b")

# GraphRAG generation context window -- federated context merged across
# several UU indexes can reach 100k+ chars; too small a num_ctx makes the
# model silently lose/confuse evidence (see tests/TA/history/masalah.txt).
GRAPH_NUM_CTX = 131072

# NaiveRAG: PDF sumber di-index on-the-fly
# CATATAN: ada trailing space di nama folder disk → "Data untuk Indexing "
NAIVE_DATASET_ROOT = ROOT_DIR / "dataset" / "TA" / "Data untuk Indexing "

# GraphRAG: working dir yang sudah diindex sebelumnya
GRAPH_KB_ROOTS = {
    "Pelayaran": ROOT_DIR / "KnowledgeBase" / "indexingPelayaran",
    "Perikanan": ROOT_DIR / "KnowledgeBase" / "indexingPerikanan",
}

# Domain hints -- identik dengan DATASET_DEFAULTS di tests/TA/Step_1_QA_ollama.py,
# supaya jawaban GraphRAG web app konsisten metodologinya dengan eksperimen federated.
DOMAIN_HINTS = {
    "Perikanan": (
        "PRIORITASKAN UU No. 31 Tahun 2004 jo. UU No. 45 Tahun 2009 tentang Perikanan "
        "sebagai dasar hukum utama untuk setiap klaim. Gunakan UU/peraturan lain (KUHP, KUHAP, UU "
        "Darurat No. 12 Tahun 1951, Permen KP, dst.) HANYA sebagai pelengkap jika UU Perikanan tidak "
        "secara spesifik mengatur hal yang ditanyakan -- jangan jadikan UU lain sebagai dasar hukum "
        "utama hanya karena kata kunci di pertanyaan (mis. 'bahan peledak') lebih mirip isi UU lain "
        "itu secara leksikal."
    ),
    "Pelayaran": (
        "PRIORITASKAN UU No. 17 Tahun 2008 tentang Pelayaran sebagai dasar hukum utama "
        "untuk setiap klaim. Gunakan UU/peraturan lain (KUHP, KUHPerdata, Permenaker, dst.) HANYA "
        "sebagai pelengkap jika UU Pelayaran tidak secara spesifik mengatur hal yang ditanyakan -- "
        "jangan jadikan UU lain sebagai dasar hukum utama hanya karena kata kunci di pertanyaan lebih "
        "mirip isi UU lain itu secara leksikal."
    ),
}
DEFAULT_RESPONSE_TYPE = (
    "Jawab mengalir dan ringkas, umumnya 2-5 kalimat, boleh lebih jika skenario "
    "melibatkan beberapa dasar hukum sekaligus."
)

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared embedding (lazy, cached) ──────────────────────────────────────────
@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

@lru_cache(maxsize=1)
def get_embed_model():
    return AutoModel.from_pretrained(EMBEDDING_MODEL)

def _hf_embed(texts):
    return hf_embed(texts, tokenizer=get_tokenizer(), embed_model=get_embed_model())

def _make_embedding_func():
    return EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=_hf_embed,
    )

# ── NaiveRAG (FAISS in-memory) ────────────────────────────────────────────────
class RAGSystem:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder      = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks: List[str] = []
        self.index         = None
        self.is_ready      = False
        self.source_label  = ""

    def _chunk(self, text: str) -> List[str]:
        chunks, start = [], 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def setup(self, text: str, label: str = "") -> int:
        self.chunks       = self._chunk(text)
        self.source_label = label
        if not self.chunks:
            self.is_ready = False
            return 0
        embs = self.embedder.encode(self.chunks)
        faiss.normalize_L2(embs)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs.astype("float32"))
        self.is_ready = True
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if not self.is_ready:
            return []
        qe = self.embedder.encode([query])
        faiss.normalize_L2(qe)
        _, idxs = self.index.search(qe.astype("float32"), top_k)
        return [self.chunks[i] for i in idxs[0] if i < len(self.chunks)]

rag_system = RAGSystem()

# ── Ollama LLM (NaiveRAG prompt) ──────────────────────────────────────────────
class OllamaLLM:
    def __init__(self):
        os.environ["OLLAMA_HOST"] = OLLAMA_HOST

    async def generate(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            resp = await asyncio.to_thread(
                ollama.generate,
                model=OLLAMA_MODEL,
                prompt=prompt,
                options={"num_predict": max_tokens, "temperature": 0.0,
                         "top_p": 0.9, "num_ctx": 32768},
            )
            return resp["response"].strip()
        except Exception as e:
            return f"⚠️ Error dari Ollama: {e}"

llm = OllamaLLM()

# ── GraphRAG instance cache ───────────────────────────────────────────────────
_graph_rag_cache: dict[str, MiniRAG] = {}

def get_graph_rag(working_dir: Path) -> MiniRAG:
    key = str(working_dir.resolve())
    if key not in _graph_rag_cache:
        _graph_rag_cache[key] = MiniRAG(
            working_dir=key,
            llm_model_func=ollama_model_complete,
            llm_model_max_token_size=32768,
            llm_model_name=OLLAMA_MODEL,
            llm_model_max_async=1,          # wajib untuk mencegah degradasi latency
            llm_model_kwargs={
                "host": OLLAMA_HOST,
                "options": {"temperature": 0.0, "num_ctx": GRAPH_NUM_CTX},
            },
            embedding_func=_make_embedding_func(),
        )
    return _graph_rag_cache[key]

# ── Path helpers ──────────────────────────────────────────────────────────────
def list_naive_pdfs(folder: str) -> List[str]:
    root = NAIVE_DATASET_ROOT / folder
    if not root.exists():
        raise HTTPException(404, f"Folder dataset '{folder}' tidak ditemukan di {root}")
    return sorted(p.name for p in root.iterdir()
                  if p.is_file() and p.suffix.lower() == ".pdf")

def list_graph_kbs(folder: str) -> List[str]:
    root = GRAPH_KB_ROOTS.get(folder)
    if root is None:
        raise HTTPException(404, f"Folder GraphRAG '{folder}' tidak dikenali")
    if not root.exists():
        raise HTTPException(404, f"Folder KnowledgeBase '{folder}' belum tersedia di {root}")
    return sorted(d.name for d in root.iterdir() if d.is_dir())

def safe_pdf_path(folder: str, filename: str) -> Path:
    root = (NAIVE_DATASET_ROOT / folder).resolve()
    p    = (root / filename).resolve()
    if root not in p.parents and p != root:
        raise HTTPException(400, "Path PDF tidak valid")
    if not p.exists():
        raise HTTPException(404, f"PDF '{filename}' tidak ditemukan")
    return p

def safe_graph_kb_path(folder: str, kb_name: str) -> Path:
    root = GRAPH_KB_ROOTS.get(folder)
    if root is None:
        raise HTTPException(404, f"Folder '{folder}' tidak dikenali")
    p = (root / kb_name).resolve()
    if not p.exists() or not p.is_dir():
        raise HTTPException(404, f"Knowledge Base '{kb_name}' tidak ditemukan")
    return p

def extract_pdf_text(path: Path) -> str:
    text = ""
    with path.open("rb") as f:
        for page in PyPDF2.PdfReader(f).pages:
            text += (page.extract_text() or "") + "\n"
    return text.strip()

# ── Auto-detect GraphRAG KB ───────────────────────────────────────────────────
_PELAYARAN_HINTS = [
    "pelayaran","kapal","maritim","laut","pelabuhan",
    "syahbandar","nakhoda","awak kapal","perkapalan","shipping",
]
_PERIKANAN_HINTS = [
    "perikanan","ikan","nelayan","budidaya","tangkap",
    "penangkapan","kapal ikan","fishing","hasil perikanan",
]

def _score_kb(query_norm: str, root_name: str, kb_name: str) -> int:
    hints = _PELAYARAN_HINTS if root_name.lower() == "pelayaran" else _PERIKANAN_HINTS
    score = sum(5 for h in hints if h in query_norm)
    if root_name.lower() in query_norm:
        score += 12
    for token in re.split(r"[_\s\-]+", kb_name.lower()):
        if token and token in query_norm:
            score += 4
    for digit in re.findall(r"\d+", kb_name):
        if digit in query_norm:
            score += 8
    return score

def auto_detect_graph_kbs(query: str, top_n: int = 1) -> List[Path]:
    """
    Return the top_n best-matching pre-indexed KB paths.
    If all scores == 0, fall back to returning all available KBs.
    """
    query_norm = re.sub(r"[^a-z0-9 ]+", " ", query.lower())
    candidates = []
    for root_name, root_path in GRAPH_KB_ROOTS.items():
        if not root_path.exists():
            continue
        for d in sorted(root_path.iterdir()):
            if d.is_dir():
                candidates.append((root_name, d.name, d,
                                   _score_kb(query_norm, root_name, d.name)))
    if not candidates:
        raise HTTPException(404, "Belum ada Knowledge Base GraphRAG yang tersedia")
    candidates.sort(key=lambda x: x[3], reverse=True)
    best_score = candidates[0][3]
    if best_score == 0:
        # Tidak bisa mendeteksi — kembalikan semua (akan di-query satu per satu)
        return [c[2] for c in candidates]
    return [c[2] for c in candidates[:top_n]]

def _domain_of(kb_path: Path) -> Optional[str]:
    """Which GRAPH_KB_ROOTS domain a KB path belongs to (for the domain hint)."""
    resolved = kb_path.resolve()
    for domain, root in GRAPH_KB_ROOTS.items():
        if root.resolve() in resolved.parents:
            return domain
    return None

# ── Federated GraphRAG answer (mirrors tests/TA/Step_1_QA_ollama.py) ──────────

class _DummyHashKV:
    def __init__(self, model_name):
        self.global_config = {"llm_model_name": model_name}

def dedupe_context_blocks(contexts: List[str]) -> str:
    """Merge -----Entities----- / -----Sources----- context blocks from multiple
    indexes, deduplicating rows by their first column (entity name or chunk_id).
    Copied from tests/TA/Step_1_QA_ollama.py (kept local -- that script isn't
    import-safe, it runs argparse/model setup at module scope)."""
    all_entities, all_chunks = [], []
    for ctx in contexts:
        if not ctx or ctx == "Error":
            continue
        ent_match = re.search(r"-----Entities-----\s*```csv\s*(.*?)\s*```", ctx, re.DOTALL)
        if ent_match:
            all_entities.extend(ent_match.group(1).strip().split("\n"))
        src_match = re.search(r"-----Sources-----\s*```csv\s*(.*?)\s*```", ctx, re.DOTALL)
        if src_match:
            all_chunks.extend(src_match.group(1).strip().split("\n"))

    seen_ent, unique_ent = set(), []
    for line in all_entities:
        key = line.split(",")[0] if "," in line else line
        if key not in seen_ent:
            seen_ent.add(key)
            unique_ent.append(line)

    seen_chunk, unique_chunk = set(), []
    for line in all_chunks:
        key = line.split(",")[0] if "," in line else line
        if key not in seen_chunk:
            seen_chunk.add(key)
            unique_chunk.append(line)

    return (
        "-----Entities-----\n```csv\n" + "\n".join(unique_ent)
        + "\n```\n-----Sources-----\n```csv\n" + "\n".join(unique_chunk) + "\n```"
    )

async def retrieve_context_from_kb(rag: MiniRAG, query: str, top_k: int) -> str:
    """Ambil context saja dari satu KB (only_need_context=True) -- retrieval murni, tanpa LLM."""
    qparam = QueryParam(mode="mini", only_need_context=True, top_k=top_k)
    try:
        ctx = await rag.aquery(query, param=qparam)
        return ctx if isinstance(ctx, str) else ""
    except Exception as e:
        return f"[Error retrieving context: {e}]"

async def graphrag_federated_answer(
    kb_paths: List[Path], query: str, top_k: int = 60,
) -> tuple[str, List[str], str]:
    """
    Federated GraphRAG generation, satu mekanisme untuk 1 atau banyak KB
    sekaligus -- persis meniru gather_federated_answer() di
    tests/TA/Step_1_QA_ollama.py: retrieval context per-KB (paralel secara
    logis, tanpa panggilan LLM per-KB) -> dedupe_context_blocks() -> SATU
    panggilan LLM dengan PROMPTS["rag_response"] (prompt produksi) + domain
    hint, bukan prompt ad-hoc/hand-written.
    Return: (answer, kb_labels, context_raw)
    """
    kb_labels = [p.name for p in kb_paths]

    contexts = await asyncio.gather(
        *[retrieve_context_from_kb(get_graph_rag(p), query, top_k) for p in kb_paths]
    )
    merged_context = dedupe_context_blocks(list(contexts))

    domains = {d for d in (_domain_of(p) for p in kb_paths) if d}
    domain_hint = "\n".join(DOMAIN_HINTS[d] for d in sorted(domains)) if domains else ""
    full_response_type = f"{domain_hint}\n{DEFAULT_RESPONSE_TYPE}" if domain_hint else DEFAULT_RESPONSE_TYPE

    system_prompt = PROMPTS["rag_response"].format(
        context_data=merged_context, response_type=full_response_type
    )

    answer = await ollama_model_complete(
        query,
        system_prompt=system_prompt,
        hashing_kv=_DummyHashKV(OLLAMA_MODEL),
        host=OLLAMA_HOST,
        options={"temperature": 0.0, "num_ctx": GRAPH_NUM_CTX},
    )
    return answer, kb_labels, merged_context

# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query:          str
    mode:           str          = "naive"
    # NaiveRAG: agar backend bisa rebuild jika cold-restart
    kb_folder:      Optional[str]       = None
    kb_filenames:   Optional[List[str]] = None
    # GraphRAG: list path absolut yang dipilih user; None → auto-detect
    graph_kb_paths: Optional[List[str]] = None

class NaiveIndexRequest(BaseModel):
    folder:    str
    filenames: List[str]

class GraphSelectRequest(BaseModel):
    folder:   str
    kb_names: List[str]   # ← sekarang list, bisa multi-select

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/dataset/list")
async def dataset_list(folder: str, mode: str = "naive"):
    """
    mode=naive → list PDF di NAIVE_DATASET_ROOT/<folder>
    mode=graph → list folder KB di GRAPH_KB_ROOTS[folder]
    """
    if mode == "graph":
        return {"folder": folder, "items": list_graph_kbs(folder)}
    return {"folder": folder, "items": list_naive_pdfs(folder)}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload sembarang PDF → index ke NaiveRAG in-memory."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Hanya mendukung file PDF")
    tmp = Path(f"/tmp/mc_upload_{file.filename}")
    try:
        with tmp.open("wb") as buf:
            shutil.copyfileobj(file.file, buf)
        text = extract_pdf_text(tmp)
        if not text:
            raise HTTPException(400, "PDF kosong atau teks tidak bisa diekstrak")
        n = rag_system.setup(text, label=file.filename)
        return {"message": "Berhasil memproses dokumen",
                "chunks_count": n, "source": file.filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/api/naive/index")
async def naive_index(req: NaiveIndexRequest):
    """Index satu atau lebih PDF dari folder dataset ke NaiveRAG in-memory."""
    if not req.filenames:
        raise HTTPException(400, "Pilih minimal satu PDF")
    texts = []
    for fn in req.filenames:
        p = safe_pdf_path(req.folder, fn)
        t = extract_pdf_text(p)
        if t:
            texts.append(t)
    if not texts:
        raise HTTPException(400, "Semua PDF yang dipilih kosong atau tidak bisa diekstrak")
    label = (req.filenames[0] if len(req.filenames) == 1
             else f"{len(req.filenames)} PDF · {req.folder}")
    n = rag_system.setup("\n\n".join(texts), label=label)
    return {
        "message": "Berhasil mengindeks PDF",
        "folder": req.folder,
        "filenames": req.filenames,
        "chunks_count": n,
        "source_label": rag_system.source_label,
    }


@app.post("/api/graph/select")
async def graph_select(req: GraphSelectRequest):
    """
    Validasi dan kembalikan path absolut untuk satu atau lebih KB GraphRAG.
    Frontend akan menyimpan kb_paths dan mengirimnya di setiap /api/chat.
    """
    if not req.kb_names:
        raise HTTPException(400, "Pilih minimal satu Knowledge Base")
    resolved = []
    for name in req.kb_names:
        p = safe_graph_kb_path(req.folder, name)
        resolved.append({"kb_name": name, "kb_path": str(p)})
    return {
        "message": f"{len(resolved)} Knowledge Base siap digunakan",
        "folder": req.folder,
        "selected": resolved,
        # Shortcut: list of paths for frontend to store
        "kb_paths": [r["kb_path"] for r in resolved],
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    # ── NaiveRAG ─────────────────────────────────────────────────────────────
    if req.mode == "naive":
        # Rebuild jika server cold-restart tapi frontend kirim info KB
        if not rag_system.is_ready and req.kb_folder and req.kb_filenames:
            try:
                texts = []
                for fn in req.kb_filenames:
                    t = extract_pdf_text(safe_pdf_path(req.kb_folder, fn))
                    if t:
                        texts.append(t)
                if texts:
                    label = (req.kb_filenames[0] if len(req.kb_filenames) == 1
                             else f"{len(req.kb_filenames)} PDF · {req.kb_folder}")
                    rag_system.setup("\n\n".join(texts), label=label)
            except Exception:
                pass

        if not rag_system.is_ready:
            return {
                "response": (
                    "⚠️ Knowledge Base belum dimuat.\n\n"
                    "Untuk NaiveRAG, silakan:\n"
                    "• Upload PDF di bagian Upload, atau\n"
                    "• Pilih PDF dari Dataset (klik 'Pilih dari Dataset')"
                ),
                "mode_used": "naive",
            }

        chunks  = rag_system.retrieve(req.query, top_k=3)
        context = "\n\n---\n\n".join(chunks)

        prompt = f"""Anda adalah asisten AI hukum maritim Indonesia yang ramah, profesional, dan membantu pengguna memahami informasi terkait pelayaran dan perikanan.

Gunakan konteks referensi hukum berikut untuk menjawab pertanyaan pengguna secara akurat, jelas, dan ringkas.

KONTEKS:
{context}

PERTANYAAN PENGGUNA:
{req.query}

ATURAN MENJAWAB:
- Gunakan Bahasa Indonesia yang natural dan mudah dipahami.
- Jawaban langsung ke inti pertanyaan tanpa bertele-tele.
- Jangan mengulang pertanyaan pengguna.
- Jika informasi tersedia di konteks, gunakan sebagai dasar jawaban.
- Jika konteks tidak cukup, katakan secara jujur informasi tidak ditemukan pada referensi.
- Hindari membuat informasi hukum yang tidak ada pada konteks.

FORMAT JAWABAN:
- Pertanyaan "Siapa"    → nama pihak/orang/instansi terkait.
- Pertanyaan "Apa"      → objek atau penjelasan inti.
- Pertanyaan "Kapan"    → waktu/tanggal/periode.
- Pertanyaan "Di mana"  → lokasi/tempat.
- Pertanyaan "Mengapa"  → alasan singkat dan jelas.
- Pertanyaan "Bagaimana"→ langkah/proses ringkas.
- Pertanyaan "Berapa"   → angka/jumlah yang relevan.

JAWABAN:"""

        answer = await llm.generate(prompt)
        return {
            "response":     answer,
            "context_used": chunks,           # list of chunk strings (untuk info)
            "context_raw":  context,          # gabungan string — ini yang dipakai LLM, untuk RAGAS
            "source":       rag_system.source_label,
            "mode_used":    "naive",
        }

    # ── GraphRAG ─────────────────────────────────────────────────────────────
    if req.mode == "graph":
        # Tentukan KB yang akan di-query
        if req.graph_kb_paths:
            # User sudah pilih secara eksplisit (bisa lebih dari satu)
            kb_paths = []
            for raw in req.graph_kb_paths:
                p = Path(raw).resolve()
                if not p.exists() or not p.is_dir():
                    raise HTTPException(404, f"KB path tidak ditemukan: {raw}")
                kb_paths.append(p)
        else:
            # Auto-detect berdasarkan keyword query
            kb_paths = auto_detect_graph_kbs(req.query, top_n=1)

        try:
            answer, kb_labels, context_raw = await graphrag_federated_answer(kb_paths, req.query)
        except Exception as e:
            raise HTTPException(500, f"GraphRAG error: {e}")

        return {
            "response":        answer,
            "context_raw":     context_raw,   # context asli yang dipakai LLM — untuk RAGAS
            "knowledge_bases": kb_labels,
            "knowledge_base":  ", ".join(kb_labels),
            "kb_paths":        [str(p) for p in kb_paths],
            "mode_used":       "graph",
        }

    raise HTTPException(400, f"Mode '{req.mode}' tidak dikenali. Gunakan 'naive' atau 'graph'.")


# ── RAGAS Dataset Ground Truth ────────────────────────────────────────────────

import unicodedata

# Paths ke CSV ground truth (sesuai lokasi di server)
RAGAS_CSV_PATHS = {
    "Pelayaran": ROOT_DIR / "dataset" / "TA" / "Palayaran Fix.csv",
    "Perikanan":  ROOT_DIR / "dataset" / "TA" / "Perikanan Fix.csv",
}

# Cache dataset agar tidak baca ulang setiap request
_gt_cache: dict[str, list[dict]] = {}

def _load_gt_dataset(domain: str) -> list[dict]:
    """Load ground truth dataset dari CSV, cache di memory."""
    if domain in _gt_cache:
        return _gt_cache[domain]
    path = RAGAS_CSV_PATHS.get(domain)
    if not path or not path.exists():
        return []
    import csv
    rows = []
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            q = (row.get("Question") or "").strip()
            a = (row.get("Answer") or "").strip()
            if q and a:
                rows.append({"question": q, "answer": a})
    _gt_cache[domain] = rows
    return rows

def _normalize_q(text: str) -> str:
    """Normalize teks untuk matching: lowercase, strip tanda baca, normalisasi unicode."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _token_overlap(a: str, b: str) -> float:
    """Jaccard token overlap antara dua string yang sudah dinormalisasi."""
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def find_ground_truth(query: str, threshold: float = 0.35) -> dict | None:
    """
    Cari ground truth dari semua dataset.
    Return: {question, ground_truth, dataset, score} atau None jika tidak ditemukan.
    """
    query_norm = _normalize_q(query)
    best = None
    best_score = threshold - 0.001

    for domain, path in RAGAS_CSV_PATHS.items():
        rows = _load_gt_dataset(domain)
        for row in rows:
            q_norm = _normalize_q(row["question"])
            score  = _token_overlap(query_norm, q_norm)
            if score > best_score:
                best_score = score
                best = {
                    "question":     row["question"],
                    "ground_truth": row["answer"],
                    "dataset":      domain,
                    "score":        round(score, 3),
                }
    return best


# ── RAGAS Evaluation (reuses tests/TA/ragas/*.py -- judge = RAGAS_EVAL_MODEL) ──

@lru_cache(maxsize=1)
def get_evaluator_llm():
    from ragas.llms import LangchainLLMWrapper
    from langchain_ollama import ChatOllama

    chat = ChatOllama(
        model=RAGAS_EVAL_MODEL, base_url=OLLAMA_HOST, temperature=0.0, num_ctx=GRAPH_NUM_CTX,
    )
    return LangchainLLMWrapper(chat)

@lru_cache(maxsize=1)
def get_evaluator_embeddings():
    from ragas.embeddings import LangchainEmbeddingsWrapper
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return LangchainEmbeddingsWrapper(emb)

@lru_cache(maxsize=1)
def get_extractors():
    llm_ = get_evaluator_llm()
    fe = ragas_faith.FaithfulnessExtractor(llm_callable=llm_)
    ce = ragas_context_recall.ContextRecallExtractor(llm_callable=llm_)
    ae = ragas_ans_correctness.AnswerCorrectnessExtractor(
        llm_callable=llm_, evaluator_embeddings=get_evaluator_embeddings()
    )
    return fe, ce, ae

MAX_CONTEXT_CHARS = 270000  # sama dengan tests/TA/ragas/eval_pelayaran.sh / eval_perikanan.sh


class RagasMatchRequest(BaseModel):
    query: str

class RagasEvalRequest(BaseModel):
    query:               str
    response:            str
    ground_truth:        str
    context_raw:         str             = ""   # context ASLI yang dipakai LLM (dari chat response)
    mode:                str             = "naive"
    # Fallback jika context_raw kosong (NaiveRAG cold restart)
    naive_kb_folder:     Optional[str]   = None
    naive_kb_filenames:  Optional[List[str]] = None


# ── RAGAS Endpoints ────────────────────────────────────────────────────────────

@app.post("/api/ragas/match")
async def ragas_match(req: RagasMatchRequest):
    """
    Cari ground truth yang cocok dengan query dari dataset Pelayaran/Perikanan CSV.
    Return {found, question, ground_truth, dataset, score}.
    """
    result = find_ground_truth(req.query)
    if result:
        return {"found": True, **result}
    return {"found": False}


@app.post("/api/ragas/evaluate")
async def ragas_evaluate(req: RagasEvalRequest):
    """
    Hitung tiga metrik RAGAS menggunakan RAGAS_EVAL_MODEL (gemma3:12b) sebagai
    LLM judge, lewat FaithfulnessExtractor/ContextRecallExtractor/
    AnswerCorrectnessExtractor -- KELAS PRODUKSI YANG SAMA dari
    tests/TA/ragas/, bukan reimplementasi ad-hoc. Ini menjamin metodologi
    penilaian identik dengan eksperimen RAGAS offline.

    context_raw = context yang BENAR-BENAR dipakai LLM saat menjawab, dikirim
    langsung dari frontend (disimpan dari response /api/chat). TIDAK
    menggunakan ground_truth sebagai proxy — itu tidak akurat.
    """
    context = truncate_context(req.context_raw.strip(), MAX_CONTEXT_CHARS)

    # Fallback: retrieve ulang dari NaiveRAG jika context hilang (cold restart)
    if not context and req.mode == "naive" and req.naive_kb_folder and req.naive_kb_filenames:
        try:
            retrieved = rag_system.retrieve(req.query, top_k=3)
            context   = "\n\n---\n\n".join(retrieved)
        except Exception:
            pass

    context_available = bool(context)

    sample = {
        "user_input": req.query,
        "question": req.query,
        "answer": req.response,
        "response": req.response,
        "reference": req.ground_truth,
        "retrieved_contexts": [context] if context else [],
        "contexts": [context] if context else [],
    }

    fe, ce, ae = get_extractors()

    async def _faithfulness():
        if not context_available:
            return None
        try:
            score = await fe._ascore(sample)
            return float(score)
        except Exception as e:
            print(f"[RAGAS faithfulness error] {e}")
            return None

    async def _context_recall():
        if not context_available or not req.ground_truth.strip():
            return None
        try:
            score = await ce._ascore(sample)
            return float(score)
        except Exception as e:
            print(f"[RAGAS context_recall error] {e}")
            return None

    async def _answer_correctness():
        try:
            score = await ae._ascore(sample)
            return float(score)
        except Exception as e:
            print(f"[RAGAS answer_correctness error] {e}")
            return None

    faith, cr, ac = await asyncio.gather(
        _faithfulness(), _context_recall(), _answer_correctness()
    )

    return {
        "answer_correctness": ac,
        "faithfulness":       faith,
        "context_recall":     cr,
        "context_available":  context_available,
        "context_chars":      len(context),   # untuk debug — berapa panjang context yang dipakai
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
