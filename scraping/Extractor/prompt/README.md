# LLM-Hukum: Data Augmentation & QA Generation untuk Hukum Pelayaran dan Perikanan Indonesia

Framework generasi dataset QA multi-hop berbasis skenario (KOBLEX-style) dari dokumen putusan pidana Indonesia, untuk keperluan pelatihan dan evaluasi sistem RAG (Retrieval-Augmented Generation) hukum.

---

## Gambaran Proses

```
Dokumen Putusan (PDF)
        │
        ▼
  Ekstraksi Terstruktur
  (skema_putusan/*.json)
        │
        ▼
  LLM Data Generator  ◄── prompt_koblex_scenario.prompt
        │
        ▼
  4 QA Pairs per Putusan
  (output/*.json)
        │
        ▼
  Validasi & Anotasi Manual
  (status: llm_generated → expert_revised → expert_validated)
```

---

## Struktur Direktori

```
LLM-Hukum/
├── putusan_pidana/          # Dokumen putusan asli (PDF)
├── skema_putusan/           # Hasil ekstraksi terstruktur per putusan (JSON)
├── prompts/
│   ├── prompt_original.prompt       # Prompt awal (single QA, domain perikanan)
│   └── prompt_koblex_scenario.prompt # Prompt aktif (4 QA/putusan, KOBLEX-style)
├── samples/
│   └── *_qa_expected.json   # Contoh output yang diharapkan per putusan
├── papers/                  # Referensi akademik
└── figures/                 # Ilustrasi format data
```

---

## Format Data Input: Skema Putusan (JSON)

Setiap putusan diekstrak ke dalam JSON terstruktur dengan skema berikut:

```json
{
  "hasil_putusan": {
    "nomor_putusan": "102/Pid.Sus/2019/PN Jkt.Utr.",
    "nama_pengadilan": "Pengadilan Negeri Jakarta Utara",
    "tanggal_putusan": "2019-04-08",
    "majelis_hakim": { "hakim_ketua": "...", "hakim_anggota": ["..."] },
    "detail_hukuman": { "denda": "Rp. 18.000.000,-", "pidana_pengganti": "3 bulan kurungan" }
  },
  "overview_kasus": {
    "terdakwa": [{ "nama": "...", "identitas_tambahan": "..." }],
    "dakwaan": [{ "pasal": "305 jo pasal 130 ayat (1)", "peraturan": "UU RI No. 17 Tahun 2008" }],
    "ringkasan_kejadian": { "kronologi_singkat": "...", "lokasi_kejadian": "...", "tanggal_kejadian": "..." }
  },
  "temuan_penting": {
    "pelanggaran_hukum": [{ "deskripsi_pelanggaran": "...", "referensi_peraturan": "..." }],
    "perbuatan_terdakwa": "...",
    "barang_bukti": [{ "nama_barang_bukti": "...", "sumber": "..." }]
  }
}
```

### Gap yang Diketahui pada Skema Saat Ini

Field-field berikut **belum tersedia** di skema JSON ekstraksi dan berdampak langsung pada kualitas output:

| Field yang Hilang | Dampak | Prioritas |
|---|---|---|
| `teks_pasal_rujukan` — teks lengkap pasal yang dikutip | LLM harus mengandalkan parametric knowledge untuk `reference_provisions[].text`; risiko halusinasi | **Kritis** |
| `pertimbangan_hukum` — ratio decidendi hakim per unsur | LLM tidak dapat membuat `reasoning_steps` berbasis dokumen; harus diinferensi | **Kritis** |
| `yurisprudensi_dikutip` — yurisprudensi MA yang dirujuk | Kehilangan konteks preseden hukum (mis. MA No. 1398/K/Pid/1984) | **Penting** |
| `faktor_hukuman` — hal memberatkan & meringankan | Tier `sanksi_pidana` kehilangan konteks penentuan jenis hukuman | **Penting** |
| `saksi` — nama dan keterangan singkat saksi | Tier `timeline_multi_aktor` tidak dapat dibangun | Sekunder |

---

## Format Data Output: QA Pairs (JSON)

Setiap putusan menghasilkan **4 QA pairs** dalam satu file JSON:

```json
{
  "qa_pairs": [
    {
      "id": "<nomor_putusan_singkat>-<tier>",
      "tier": "cross_pasal | analogis | sanksi_pidana | komparasi_subjek",
      "user_input_style": "skenario | analogi",
      "reasoning_depth_klasifikasi": {
        "depth": 2,
        "justifikasi": "Penjelasan per hop: pasal mana, kontribusinya, mengapa tidak bisa dihilangkan."
      },
      "background_scenario": "Narasi fakta anonim (60–150 kata).",
      "question": "Pertanyaan multi-hop gaya chatbot pengguna.",
      "answer": "Jawaban analisis hukum (120–350 kata), semua pasal disebut lengkap.",
      "reasoning_steps": ["Langkah 1: ...", "Langkah 2: ...", "Langkah N: ..."],
      "reference_provisions": [
        { "provision_id": "UU No. X Tahun Y / Pasal Z ayat (N)", "text": "teks lengkap pasal" }
      ],
      "referensi_hukum": ["Pasal X ayat (N) UU No. Y Tahun Z tentang Nama UU"]
    }
  ]
}
```

### Empat Tier QA yang Dihasilkan

| No | Tier | `user_input_style` | Reasoning Depth | Deskripsi |
|---|---|---|---|---|
| 1 | `cross_pasal` | `skenario` | 2 | Menghubungkan pasal kewajiban → pasal sanksi; menguji apakah pengguna memahami relasi normatif antar pasal |
| 2 | `analogis` | `analogi` | 2 | Menggunakan analogi lintas domain untuk menguji sifat absolut/kondisional kewajiban hukum |
| 3 | `sanksi_pidana` | `skenario` | 2 | Pertanyaan eksplisit soal jenis, besaran, dan sifat hukuman (penjara/denda, alternatif vs. kumulatif) |
| 4 | `komparasi_subjek` | `skenario` | 3 | Membandingkan pasal dan ancaman sanksi untuk 2–3 subjek hukum berbeda dalam satu skenario |

### Prinsip Validasi Multi-hop (RC3 / KOBLEX Partial Check)

Setiap QA pair harus lulus uji berikut sebelum dianggap valid:

> **Hapus satu pasal dari `reference_provisions`.** Jika jawaban masih bisa diberikan secara lengkap tanpa pasal tersebut, pasal itu tidak boleh ada di `reference_provisions`.

Ini memastikan setiap instance benar-benar memerlukan semua pasal yang dicantumkan — bukan sekadar listing referensi.

---

## Cara Menggunakan Prompt

### Prasyarat

```python
import json

# Muat hasil ekstraksi JSON untuk satu putusan
with open("skema_putusan/102_Pid_Sus_2019_PN_Jkt_Utr_extracted.json") as f:
    extracted_data = json.load(f)

# Muat prompt template
exec(open("prompts/prompt_koblex_scenario.prompt").read())
# Variabel `prompt` sekarang berisi string prompt yang siap dikirim ke LLM
```

### Pengiriman ke LLM

```python
# Contoh dengan OpenAI-compatible API
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
    temperature=0.7
)

result = json.loads(response.choices[0].message.content)
# result["qa_pairs"] berisi 4 item
```

### Output yang Diharapkan

Lihat `samples/102_Pid_Sus_2019_PN_Jkt_Utr_qa_expected.json` untuk contoh lengkap output 4 QA pairs dari putusan `102/Pid.Sus/2019/PN Jkt.Utr.`

---

## Aturan Kualitas Prompt (Ringkasan)

### Background Scenario (B)
- Dipisah dari pertanyaan Q — B = narasi fakta, Q = pertanyaan saja
- Semua pihak dianonimkan (tidak ada nama terdakwa/kapal asli)
- Memuat semua konteks faktual yang dibutuhkan (lokasi, kondisi, aktor, tindakan)
- 60–150 kata

### Question (Q)
- Gaya chatbot pengguna — bukan kalimat akademik
- Jika menyebut regulasi: wajib nomor lengkap (pasal + UU + tahun)
- Memerlukan minimal 2 pasal dari `reference_provisions` untuk dijawab

### Answer (A)
- Semua referensi hukum: nomor pasal + nama UU + tahun (bukan singkatan)
- Tier `sanksi_pidana`: wajib sebut angka maksimum + sifat alternatif/kumulatif
- Tier `komparasi_subjek`: wajib format per-subjek bernomor, bukan satu paragraf gabungan
- 120–350 kata

### Reference Provisions (C)
- Teks lengkap pasal — bukan hanya nomornya
- Minimal 2 pasal, masing-masing membahas aspek berbeda
- Setiap pasal harus benar-benar diperlukan (lulus RC3 partial check)

---

## Rekomendasi Field Tambahan untuk Versi Berikutnya

Field berikut direkomendasikan untuk ditambahkan ke skema output QA, dikelompokkan berdasarkan prioritas:

### Kritis
```json
"hop_chain": [
  { "hop": 1, "provision_id": "...", "kontribusi": "...", "dapat_dihilangkan": false }
],
"partial_check_passed": true,
"source_putusan": "102/Pid.Sus/2019/PN Jkt.Utr."
```

### Penting
```json
"domain_hukum": "pelayaran",
"temporal_validity": { "UU No. 17 Tahun 2008": "aktif" },
"difficulty": 2,
"answer_grounding": "full | partial | parametric"
```

### Opsional (untuk penelitian)
```json
"negation_test": "Jika nakhoda tidak mengetahui kerusakan radio...",
"validation_status": "llm_generated | expert_revised | expert_validated",
"evaluasi_manusia": {
  "fluency": null, "practicality": null, "relevance": null,
  "legal_accuracy": null, "complexity": null
}
```

---

## Referensi

- **KOBLEX**: Lee et al., *"KOBLEX: Open Legal Question Answering with Multi-hop Reasoning"*, EMNLP 2025 — metodologi utama untuk desain format QA dan validasi multi-hop.
- **UU No. 17 Tahun 2008** tentang Pelayaran — sumber utama pasal yang dirujuk dalam domain pelayaran.
- **Putusan No. 102/Pid.Sus/2019/PN Jkt.Utr.** — contoh putusan dasar untuk pengembangan prompt dan sampel output.
