"""
qa_evaluator.py
===============
Evaluator untuk dataset QA hukum pelayaran/perikanan Indonesia.

Dua layer evaluasi:
  Layer 1 — Rule-based  : tidak butuh LLM, instan, gratis
  Layer 2 — LLM-as-Judge: pakai Ollama lokal (default: gemma3:12b)
                          atau OpenAI jika set OPENAI_API_KEY

Install:
    pip install pandas requests colorama tqdm

Usage:
    # Evaluasi semua baris, kedua layer
    python qa_evaluator.py --csv data.csv

    # Hanya rule-based (cepat, tanpa LLM)
    python qa_evaluator.py --csv data.csv --no-llm

    # Pakai model Ollama berbeda
    python qa_evaluator.py --csv data.csv --model qwen3.6:27b

    # Pakai OpenAI
    python qa_evaluator.py --csv data.csv --model openai

    # EX
    python /home/mdewana/GraphRAG/MiniRAG[gemma3]/scraping/extractor/qa_result/qa_evaluator.py --csv /home/mdewana/GraphRAG/MiniRAG[gemma3]/scraping/extractor/qa_result/perikanan/gpt/batch_qa_1779142418463.csv --model openai --output hasil_
    eval.csv
    # Simpan hasil ke file
    python qa_evaluator.py --csv data.csv --output hasil_evaluasi.csv
"""

import os
import re
import json
import argparse
import textwrap
import csv
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

# ──────────────────────────────────────────────
# KONFIGURASI
# ──────────────────────────────────────────────
OLLAMA_ENDPOINT  = os.environ.get("OLLAMA_ENDPOINT",  "http://127.0.0.1:11500/api/generate")
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL    = "gemma3:12b"

# Kata kunci hukum Indonesia yang wajib ada di jawaban berkualitas
LEGAL_KEYWORDS = [
    r"\bpasal\b", r"\buu\b|\bundang.undang\b", r"\bperaturan\b",
    r"\bpidana\b", r"\bdenda\b", r"\bpenjara\b", r"\bkurungan\b",
    r"\bpugnant\b|\bmelanggar\b", r"\bterbukti\b|\bdinyatakan bersalah\b",
    r"\bmajelis hakim\b|\bpengadilan\b",
]

TIER_VALID = {"cross_pasal", "timeline_multi_aktor", "norma_kontradiksi",
              "hipotetis", "analogis", "proporsionalitas"}

PREDICATE_TYPES = {"faktual", "normatif", "prosedural", "inferensial"}


# ══════════════════════════════════════════════
# LAYER 1: RULE-BASED CHECKS
# ══════════════════════════════════════════════

def check_rule_based(row: dict) -> dict:
    scores = {}
    details = {}

    q   = str(row.get("Question", "")).strip()
    a   = str(row.get("Answer",   "")).strip()
    r   = str(row.get("Reasoning","")).strip()
    ref = str(row.get("References","")).strip()
    gr  = str(row.get("Graph_Relations","")).strip()
    tier = str(row.get("Tier","")).strip().lower()

    # ── R1: Panjang pertanyaan (ideal 30-250 char)
    qlen = len(q)
    scores["q_length_ok"] = 1.0 if 30 <= qlen <= 300 else (0.5 if qlen > 0 else 0.0)
    details["q_length"] = qlen

    # ── R2: Panjang jawaban (minimal 120 kata sesuai prompt)
    a_words = len(a.split())
    scores["a_length_ok"] = 1.0 if a_words >= 120 else (a_words / 120)
    details["a_word_count"] = a_words

    # ── R3: Kehadiran kata kunci hukum dalam jawaban
    kw_found = sum(1 for kw in LEGAL_KEYWORDS if re.search(kw, a.lower()))
    scores["a_legal_keywords"] = round(kw_found / len(LEGAL_KEYWORDS), 2)
    details["legal_kw_found"] = kw_found

    # ── R4: Referensi hukum tidak kosong & ada nomor pasal/uu
    has_ref = bool(ref) and ref != "nan"
    has_pasal = bool(re.search(r"pasal\s*\d+|uu\s*no|nomor\s*\d+|tahun\s*\d{4}", ref.lower()))
    scores["ref_quality"]   = 1.0 if (has_ref and has_pasal) else (0.5 if has_ref else 0.0)
    details["ref_snippet"]  = ref[:80] if has_ref else "(kosong)"

    # ── R5: Reasoning minimal 2 langkah (split by ' | ' atau newline)
    steps = [s.strip() for s in re.split(r"\s*\|\s*|\n", r) if s.strip()]
    scores["reasoning_steps"] = 1.0 if len(steps) >= 2 else (0.5 if len(steps) == 1 else 0.0)
    details["step_count"] = len(steps)

    # ── R6: Graph relations — validasi JSON & field wajib
    gr_score, gr_detail = _validate_graph_relations(gr)
    scores["graph_relations_valid"] = gr_score
    details["graph_detail"] = gr_detail

    # ── R7: Tier valid
    scores["tier_valid"] = 1.0 if tier in TIER_VALID else 0.0
    details["tier"] = tier

    # ── R8: Self-contained question — pertanyaan tidak merujuk "dokumen ini" dll
    leak_words = r"dokumen ini|putusan ini|berkas ini|file ini|terdakwa di atas"
    scores["q_self_contained"] = 0.0 if re.search(leak_words, q.lower()) else 1.0

    # ── R9: Jawaban tidak echo pertanyaan secara verbatim (panjang overlap)
    q_tokens = set(q.lower().split())
    a_tokens = set(a.lower().split())
    overlap  = len(q_tokens & a_tokens) / max(len(q_tokens), 1)
    scores["a_not_echo_q"] = 1.0 if overlap < 0.6 else round(1.0 - overlap, 2)
    details["q_a_overlap"] = round(overlap, 2)

    # ── R10: Inferensial node hadir di graph relations
    scores["has_inferensial"] = _check_predicate_type(gr, "inferensial")

    # Skor total rule-based (rata-rata semua dimensi)
    total = round(sum(scores.values()) / len(scores), 3)

    return {
        "rb_total": total,
        "rb_scores": scores,
        "rb_details": details,
    }


def _validate_graph_relations(gr_str: str) -> tuple:
    if not gr_str or gr_str == "nan":
        return 0.0, "kosong"
    try:
        data = json.loads(gr_str)
        if not isinstance(data, list) or len(data) == 0:
            return 0.2, "list kosong"
        required = {"subject", "predicate", "object"}
        n_valid = sum(1 for item in data if required.issubset(item.keys()))
        n_has_type = sum(1 for item in data
                         if item.get("predicate_type","").lower() in PREDICATE_TYPES)
        count_score  = min(1.0, len(data) / 4)           # ideal >= 4 node
        field_score  = n_valid / len(data)
        type_score   = n_has_type / len(data)
        total = round((count_score + field_score + type_score) / 3, 2)
        return total, f"{len(data)} nodes, {n_valid} valid, {n_has_type} typed"
    except json.JSONDecodeError as e:
        return 0.1, f"JSON error: {e}"


def _check_predicate_type(gr_str: str, ptype: str) -> float:
    try:
        data = json.loads(gr_str)
        has = any(str(item.get("predicate_type","")).lower() == ptype for item in data)
        return 1.0 if has else 0.0
    except Exception:
        return 0.0


# ══════════════════════════════════════════════
# LAYER 2: LLM-AS-JUDGE
# ══════════════════════════════════════════════

LLM_JUDGE_PROMPT = """Kamu adalah hakim kualitas dataset QA hukum Indonesia. 
Nilailah pasangan QA berikut dalam skala 0.0–1.0 untuk 5 dimensi.

PERTANYAAN:
{question}

JAWABAN:
{answer}

REFERENSI HUKUM:
{references}

GRAPH RELATIONS:
{graph_relations}

Berikan penilaian HANYA dalam format JSON berikut (tanpa teks lain):
{{
  "faithfulness":       <0.0-1.0>,  // Apakah jawaban konsisten secara faktual dengan referensi hukum?
  "answer_relevancy":   <0.0-1.0>,  // Seberapa relevan jawaban dengan pertanyaan?
  "legal_accuracy":     <0.0-1.0>,  // Apakah pasal/UU yang disebutkan tepat dan lengkap?
  "reasoning_quality":  <0.0-1.0>,  // Apakah alur penalaran logis dan multi-hop?
  "graph_consistency":  <0.0-1.0>,  // Apakah graph relations konsisten dengan isi jawaban?
  "brief_critique":     "<kalimat singkat penilaian kelemahan utama>"
}}"""


def call_llm_judge(row: dict, model: str) -> dict:
    prompt = LLM_JUDGE_PROMPT.format(
        question       = row.get("Question",""),
        answer         = row.get("Answer",""),
        references     = row.get("References",""),
        graph_relations= row.get("Graph_Relations",""),
    )

    try:
        if model == "openai":
            return _call_openai(prompt)
        else:
            return _call_ollama(prompt, model)
    except Exception as e:
        return {"error": str(e), "llm_total": 0.0}


def _call_ollama(prompt: str, model: str) -> dict:
    resp = requests.post(
        OLLAMA_ENDPOINT,
        json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
        timeout=180
    )
    resp.raise_for_status()
    text = resp.json().get("response","").strip()
    return _parse_judge_json(text)


def _call_openai(prompt: str) -> dict:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY tidak di-set")
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        },
        timeout=60
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return _parse_judge_json(text)


def _parse_judge_json(text: str) -> dict:
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            text = m.group(0)
        data = json.loads(text)
        numeric_keys = ["faithfulness","answer_relevancy","legal_accuracy",
                        "reasoning_quality","graph_consistency"]
        values = [float(data.get(k, 0)) for k in numeric_keys]
        data["llm_total"] = round(sum(values) / len(values), 3)
        return data
    except Exception as e:
        return {"error": f"parse error: {e}", "llm_total": 0.0, "raw": text[:200]}


# ══════════════════════════════════════════════
# MAIN EVALUATOR
# ══════════════════════════════════════════════

def evaluate_csv(
    csv_path: str,
    use_llm: bool = True,
    model: str = DEFAULT_MODEL,
    output_path: str = None,
    max_rows: int = None,
) -> pd.DataFrame:

    df = pd.read_csv(csv_path)
    print(f"\n{Fore.CYAN}{'═'*60}")
    print(f"  QA EVALUATOR — Hukum Pelayaran/Perikanan Indonesia")
    print(f"{'═'*60}{Style.RESET_ALL}")
    print(f"  File  : {csv_path}")
    print(f"  Rows  : {len(df)}{' (limited to ' + str(max_rows) + ')' if max_rows else ''}")
    print(f"  LLM   : {'OFF' if not use_llm else model}")
    print()

    if max_rows:
        df = df.head(max_rows)

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        row_dict = row.to_dict()
        result   = {"Filename": row_dict.get("Filename",""), "Tier": row_dict.get("Tier","")}

        # ── Layer 1: Rule-based
        rb = check_rule_based(row_dict)
        result["rb_total"]              = rb["rb_total"]
        result["rb_q_length_ok"]        = rb["rb_scores"]["q_length_ok"]
        result["rb_a_length_ok"]        = rb["rb_scores"]["a_length_ok"]
        result["rb_a_legal_keywords"]   = rb["rb_scores"]["a_legal_keywords"]
        result["rb_ref_quality"]        = rb["rb_scores"]["ref_quality"]
        result["rb_reasoning_steps"]    = rb["rb_scores"]["reasoning_steps"]
        result["rb_graph_valid"]        = rb["rb_scores"]["graph_relations_valid"]
        result["rb_tier_valid"]         = rb["rb_scores"]["tier_valid"]
        result["rb_q_self_contained"]   = rb["rb_scores"]["q_self_contained"]
        result["rb_has_inferensial"]    = rb["rb_scores"]["has_inferensial"]
        result["detail_a_words"]        = rb["rb_details"]["a_word_count"]
        result["detail_graph"]          = rb["rb_details"]["graph_detail"]

        # ── Layer 2: LLM Judge
        if use_llm:
            llm = call_llm_judge(row_dict, model)
            result["llm_total"]             = llm.get("llm_total", 0.0)
            result["llm_faithfulness"]      = llm.get("faithfulness", "-")
            result["llm_answer_relevancy"]  = llm.get("answer_relevancy", "-")
            result["llm_legal_accuracy"]    = llm.get("legal_accuracy", "-")
            result["llm_reasoning_quality"] = llm.get("reasoning_quality", "-")
            result["llm_graph_consistency"] = llm.get("graph_consistency", "-")
            result["llm_critique"]          = llm.get("brief_critique", "")
            result["llm_error"]             = llm.get("error", "")

            # Skor gabungan (60% LLM + 40% rule-based)
            result["final_score"] = round(
                0.6 * float(llm.get("llm_total", 0)) +
                0.4 * rb["rb_total"], 3
            )
        else:
            result["final_score"] = rb["rb_total"]

        results.append(result)

    results_df = pd.DataFrame(results)

    # ── Print summary
    _print_summary(results_df, use_llm)

    # ── Save
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\n{Fore.GREEN}✓ Hasil disimpan: {output_path}{Style.RESET_ALL}")

    return results_df


def _print_summary(df: pd.DataFrame, use_llm: bool):
    print(f"\n{Fore.CYAN}{'─'*60}")
    print(f"  HASIL EVALUASI SUMMARY")
    print(f"{'─'*60}{Style.RESET_ALL}")

    cols_rb = [c for c in df.columns if c.startswith("rb_") and c != "rb_total"]

    print(f"\n{Fore.YELLOW}  ── RULE-BASED (Layer 1){Style.RESET_ALL}")
    print(f"  {'Dimensi':<35} {'Avg':>6}  {'Min':>6}  {'Max':>6}")
    print(f"  {'─'*55}")
    for col in ["rb_total"] + cols_rb:
        try:
            avg = df[col].astype(float).mean()
            mn  = df[col].astype(float).min()
            mx  = df[col].astype(float).max()
            color = Fore.GREEN if avg >= 0.7 else (Fore.YELLOW if avg >= 0.4 else Fore.RED)
            label = col.replace("rb_","").replace("_"," ")
            print(f"  {label:<35} {color}{avg:>6.2f}{Style.RESET_ALL}  {mn:>6.2f}  {mx:>6.2f}")
        except Exception:
            pass

    if use_llm:
        llm_cols = ["llm_faithfulness","llm_answer_relevancy","llm_legal_accuracy",
                    "llm_reasoning_quality","llm_graph_consistency"]
        print(f"\n{Fore.YELLOW}  ── LLM-AS-JUDGE (Layer 2){Style.RESET_ALL}")
        print(f"  {'Dimensi':<35} {'Avg':>6}  {'Min':>6}  {'Max':>6}")
        print(f"  {'─'*55}")
        for col in ["llm_total"] + llm_cols:
            try:
                avg = pd.to_numeric(df[col], errors="coerce").dropna().mean()
                mn  = pd.to_numeric(df[col], errors="coerce").dropna().min()
                mx  = pd.to_numeric(df[col], errors="coerce").dropna().max()
                color = Fore.GREEN if avg >= 0.7 else (Fore.YELLOW if avg >= 0.4 else Fore.RED)
                label = col.replace("llm_","").replace("_"," ")
                print(f"  {label:<35} {color}{avg:>6.2f}{Style.RESET_ALL}  {mn:>6.2f}  {mx:>6.2f}")
            except Exception:
                pass

    # Per-row ranking
    print(f"\n{Fore.YELLOW}  ── PER-ROW RANKING{Style.RESET_ALL}")
    df_sorted = df.sort_values("final_score", ascending=False)
    print(f"  {'#':<4} {'Filename':<40} {'Final':>7}  Flag")
    print(f"  {'─'*65}")
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        score = float(row["final_score"])
        flag  = "✓" if score >= 0.7 else ("⚠" if score >= 0.4 else "✗")
        color = Fore.GREEN if score >= 0.7 else (Fore.YELLOW if score >= 0.4 else Fore.RED)
        fname = str(row.get("Filename",""))[:38]
        print(f"  {i+1:<4} {fname:<40} {color}{score:>6.3f}{Style.RESET_ALL}  {color}{flag}{Style.RESET_ALL}")

    # Distribusi kualitas
    total = len(df)
    good  = (pd.to_numeric(df["final_score"], errors="coerce") >= 0.7).sum()
    mid   = ((pd.to_numeric(df["final_score"], errors="coerce") >= 0.4) &
             (pd.to_numeric(df["final_score"], errors="coerce") < 0.7)).sum()
    bad   = (pd.to_numeric(df["final_score"], errors="coerce") < 0.4).sum()

    print(f"\n  {Fore.GREEN}BAIK  (≥0.7): {good}/{total}{Style.RESET_ALL}   "
          f"{Fore.YELLOW}SEDANG (0.4–0.7): {mid}/{total}{Style.RESET_ALL}   "
          f"{Fore.RED}BURUK  (<0.4): {bad}/{total}{Style.RESET_ALL}")

    print(f"\n  Overall final_score: "
          f"{Fore.CYAN}{pd.to_numeric(df['final_score'], errors='coerce').mean():.3f}{Style.RESET_ALL}\n")


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluator dataset QA hukum pelayaran/perikanan Indonesia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Contoh:
          python qa_evaluator.py --csv data.csv
          python qa_evaluator.py --csv data.csv --no-llm
          python qa_evaluator.py --csv data.csv --model qwen3.6:27b --output out.csv
          python qa_evaluator.py --csv data.csv --model openai --max-rows 10
        """)
    )
    parser.add_argument("--csv",      required=True, help="Path ke file CSV dataset")
    parser.add_argument("--no-llm",   action="store_true", help="Skip LLM judge, rule-based only")
    parser.add_argument("--model",    default=DEFAULT_MODEL, help="Model Ollama atau 'openai'")
    parser.add_argument("--output",   default=None, help="Simpan hasil ke CSV ini")
    parser.add_argument("--max-rows", type=int, default=None, help="Batasi jumlah baris dievaluasi")

    args = parser.parse_args()

    evaluate_csv(
        csv_path   = args.csv,
        use_llm    = not args.no_llm,
        model      = args.model,
        output_path= args.output,
        max_rows   = args.max_rows,
    )