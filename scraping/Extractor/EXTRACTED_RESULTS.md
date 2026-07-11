# View Extracted Results by Category

Fitur untuk melihat hasil ekstraksi JSON dari dokumen putusan yang sudah diproses, diorganisir berdasarkan kategori (pelayaran, perikanan, dll).

## Struktur Folder

```
hasil_ekstraksi/
├── pelayaran/
│   ├── 17_Pid.B_2020_PN Ran_extracted.json
│   ├── 18_Pid.B_2020_PN Dgl_extracted.json
│   ├── 26_PID_2019_PT SMR_extracted.json
│   ├── 6_PID_2019_PT SMR_extracted.json
│   └── 8_Pid.B_2020_PN Kdi_extracted.json
├── perikanan/
│   ├── 1_Pid.Sus-PRK_2018_PN Mme_extracted.json
│   ├── 1_Pid.Sus-PRK_2019_PN Jkt.Utr_extracted.json
│   ├── 1_Pid.Sus-PRK_2019_PN Lbh_extracted.json
│   ├── 1_Pid.Sus-PRK_2019_PN Mme_extracted.json
│   └── 1_Pid.Sus-PRK_2020_PN Mme_extracted.json
└── [kategori lainnya]/
```

## Cara Menggunakan via Web UI

1. Buka `http://localhost:5123`
2. **Section "Extracted Results"** akan menampilkan tab untuk setiap kategori:
   - 📋 Pelayaran (5 files)
   - 📋 Perikanan (5 files)
   - [kategori lainnya]

3. Klik tab kategori untuk melihat daftar file dalam kategori tersebut
4. Klik file untuk membukanya dan melihat preview JSON hasil ekstraksi
5. Hasil akan ditampilkan di area utama dengan:
   - Formatted JSON dengan syntax highlighting
   - Button untuk download JSON
   - Button untuk **✨ Generate QA** dari ekstraksi tersebut

## API Endpoints

### List Categories
```bash
GET http://localhost:5123/results/categories
```

Response:
```json
[
  {
    "name": "pelayaran",
    "count": 5
  },
  {
    "name": "perikanan",
    "count": 5
  }
]
```

### List Files by Category
```bash
GET http://localhost:5123/results/by-category/pelayaran
```

Response:
```json
[
  {
    "name": "6_PID_2019_PT SMR_extracted.json",
    "filename": "6_PID_2019_PT SMR",
    "category": "pelayaran",
    "mtime": 1672531200,
    "size": 8192
  },
  ...
]
```

### Get File from Category
```bash
GET http://localhost:5123/results/file-by-category/pelayaran/6_PID_2019_PT%20SMR_extracted.json
```

Response: Full JSON data dari file

## Fitur

✅ **Kategori Otomatis**: Sistem otomatis mendeteksi folder kategori yang ada di `hasil_ekstraksi/`

✅ **Tab Navigation**: Tab untuk setiap kategori dengan jumlah file di dalamnya

✅ **File List**: Daftar file dalam kategori dengan ukuran file

✅ **JSON Preview**: Preview hasil ekstraksi dengan formatted JSON dan syntax highlighting

✅ **QA Generation**: Generate skenario QA dari hasil ekstraksi yang dipilih

✅ **Download**: Download file JSON yang dipilih

## Workflow Lengkap

1. **Upload & Extract**: Extract dokumen PDF baru via form upload
   - Hasil disimpan ke `results/` (untuk file baru)
   - Preview langsung di area utama

2. **Browse Existing**: Lihat hasil ekstraksi yang sudah ada
   - Klik tab kategori (Pelayaran/Perikanan)
   - Pilih file dari daftar
   - Preview JSON hasil ekstraksi

3. **Generate QA**: Buat skenario QA dari ekstraksi
   - Klik tombol **✨ Generate QA**
   - Tunggu proses generasi (1-3 menit)
   - Lihat hasil → Download

4. **Download Results**: Export hasil ekstraksi atau QA
   - Download JSON ekstraksi
   - Download QA dalam format txt

## Catatan Teknis

- Kategori dideteksi otomatis dari folder di `hasil_ekstraksi/`
- Hanya folder dengan file `*_extracted.json` yang ditampilkan
- File diurutkan berdasarkan modification time (terbaru terlebih dahulu)
- Ukuran file ditampilkan dalam KB

## Troubleshooting

### "No extraction results in this category"

- Folder kategori kosong atau tidak ada file `*_extracted.json`
- Pastikan struktur folder benar di `hasil_ekstraksi/`

### Kategori tidak muncul

- Folder kategori harus langsung di dalam `hasil_ekstraksi/`
- Folder harus mengandung setidaknya satu file `*_extracted.json`
- Refresh browser atau klik tombol ⟳ (refresh categories)

### File tidak bisa dibuka

- Pastikan file adalah JSON valid
- Periksa permissions folder `hasil_ekstraksi/`
- Lihat browser console untuk error detail

## App Backend dan Web UI

Folder ini juga mendokumentasikan penggunaan `scraping/Extractor/app.py` dan `scraping/Extractor/landingai_extractor.html`.

### Menjalankan backend

1. Masuk ke folder `scraping/Extractor`
2. Jalankan:
```bash
python app.py
```
3. Buka `http://localhost:5123`

### Fitur utama

- Upload PDF putusan dan kirim ke LandingAI untuk ekstraksi schema
- Menyimpan hasil ekstraksi ke folder `results/`
- Memuat hasil ekstraksi yang sudah ada dari folder `hasil_ekstraksi/`
- Generate QA dari hasil ekstraksi dengan pilihan model dan prompt

### Endpoint backend

- `GET /health`
- `POST /extract`
- `POST /generate-qa`
- `GET /results/list`
- `GET /results/file/<name>`
- `GET /results/categories`
- `GET /results/by-category/<category>`
- `GET /results/file-by-category/<category>/<name>`

### Catatan environment

- Set `OPENAI_API_KEY` untuk model `openai/chatgpt`
- Set `OLLAMA_ENDPOINT` jika menggunakan Ollama di alamat selain `http://127.0.0.1:11500/api/generate`
- Jangan commit file `.env` yang berisi kunci API ke repository publik

## Catatan Teknis Tambahan

- Hasil ekstraksi batch disimpan di `results/`
- Hasil ekstraksi kategori yang sudah tersedia dibaca dari `hasil_ekstraksi/`
- UI web ada di `landingai_extractor.html`
