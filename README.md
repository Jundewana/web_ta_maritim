# web_ta_maritim

Lampiran kode (source code appendix) untuk Tugas Akhir/Skripsi oleh **Muhamad Arjun Dewana**, membahas perbandingan **MiniRAG (GraphRAG)** dan **Naive RAG** untuk sistem tanya-jawab hukum maritim Indonesia (domain Perikanan & Pelayaran).

Repo ini berisi dua bagian dari pipeline penelitian:

| Folder | Isi |
|---|---|
| [`web/`](web/) | Aplikasi web chatbot tanya-jawab hukum (frontend React + backend FastAPI) yang menjalankan MiniRAG (GraphRAG) dan Naive RAG secara langsung, lengkap dengan panel evaluasi RAGAS (Faithfulness, Context Recall, Answer Correctness) — dipakai untuk demo sistem di sidang. |
| [`scraping/Extractor/`](scraping/Extractor/) | Tool ekstraksi dataset QA dari dokumen putusan pengadilan (PDF) hukum Perikanan & Pelayaran, memakai LandingAI ADE untuk parsing dokumen dan LLM (Ollama/OpenAI) untuk generate pasangan QA gaya KOBLEX — dipakai untuk membangun dataset evaluasi 100 soal per domain. |

## Status: kode pendukung, bukan repo yang bisa langsung `run` berdiri sendiri

Repo ini adalah **snapshot dari dua subfolder** milik repo penelitian utama (monorepo MiniRAG yang sudah dimodifikasi untuk Bahasa Indonesia). Tujuannya sebagai lampiran yang menunjukkan implementasi kode secara utuh, **bukan** sebagai paket yang siap dijalankan begitu saja, karena `web/backend/main.py` masih bergantung pada beberapa hal yang **tidak** ikut disertakan di sini (ukuran besar / bukan cakupan lampiran):

- Package inti `minirag/` (core library GraphRAG yang sudah dimodifikasi — prompt Bahasa Indonesia, entity schema hukum, dsb.)
- Knowledge graph yang sudah ter-index (`KnowledgeBase/indexingPerikanan/`, `KnowledgeBase/indexingPelayaran/`)
- Modul evaluasi RAGAS (`tests/TA/ragas/ragas_faith.py`, `ragas_context_recall.py`, `ragas_ans_correctness.py`, `eval_ragas.py`)
- Server Ollama lokal (model `gemma3:12b`) untuk generasi jawaban maupun LLM-as-judge

Untuk menjalankan `web/` secara fungsional, folder ini perlu diletakkan kembali di dalam struktur repo penelitian utama (lihat `web/run_web.sh` untuk contoh skrip SBATCH yang dipakai saat demo).

## Konfigurasi & API key

Tidak ada API key yang disertakan di repo ini. Kedua aplikasi butuh file `.env` sendiri (tidak di-commit, lihat `.gitignore`) berisi:

```text
OPENAI_API_KEY=...      # opsional, hanya dipakai scraping/Extractor untuk mode LLM=openai
LANDING_AI_API_KEY=...  # opsional, dipakai scraping/Extractor untuk parsing PDF via LandingAI ADE
```

Model default untuk menjawab & menjadi judge RAGAS adalah **`gemma3:12b`** via Ollama lokal — tidak butuh API key eksternal untuk alur utama (`web/`).

## Dokumentasi lebih detail

- [`web/frontend/README.md`](web/frontend/README.md) — setup React/Vite
- [`scraping/Extractor/README.md`](scraping/Extractor/README.md) — instalasi, endpoint, alur kerja ekstraksi & generate QA
- [`scraping/Extractor/prompt/README.md`](scraping/Extractor/prompt/README.md) — dokumentasi skema prompt QA (KOBLEX-style: cross_pasal, analogis, sanksi_pidana, komparasi_subjek)
