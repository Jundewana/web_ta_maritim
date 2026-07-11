# Agentic Document Extractor

Dokumentasi untuk `scraping/Extractor/app.py` dan `scraping/Extractor/landingai_extractor.html`.

## Tujuan

Aplikasi ini menyediakan backend Flask untuk:

- Mengunggah PDF putusan
- Mengirim PDF ke LandingAI ADE untuk parsing dan ekstraksi schema
- Menyimpan hasil ekstraksi ke folder `results/`
- Menyajikan hasil ekstraksi kategori yang sudah ada melalui `hasil_ekstraksi/`
- Menghasilkan QA skenario dari hasil ekstraksi menggunakan model Ollama atau OpenAI

## Instalasi

```bash
cd scraping/Extractor
python -m pip install -r ../requirements.txt
```

> File `requirements.txt` utama berada di folder `scraping/`.

## Menjalankan backend

```bash
cd scraping/Extractor
python app.py
```

Lalu buka di browser:

```bash
http://localhost:5123
```

## Konfigurasi environment

Gunakan file `.env` atau environment variable:

- `OPENAI_API_KEY` untuk model `openai/chatgpt`
- `OLLAMA_ENDPOINT` untuk alamat endpoint Ollama jika tidak menggunakan default

Contoh `.env`:

```text
OPENAI_API_KEY=sk-...
OLLAMA_ENDPOINT=http://127.0.0.1:11500/api/generate
```

> Jangan commit `.env` yang berisi kunci API ke repository publik.

## Endpoint utama

- `GET /health`
- `GET /results/list`
- `GET /results/file/<name>`
- `GET /results/categories`
- `GET /results/by-category/<category>`
- `GET /results/file-by-category/<category>/<name>`
- `POST /extract`
- `POST /generate-qa`

## UI Web

File `landingai_extractor.html` menyediakan antarmuka berikut:

- Input backend URL
- API key LandingAI
- Pilihan model QA
- Pilihan prompt QA
- Upload PDF dan schema JSON
- Daftar file hasil ekstraksi batch
- Panel kategori hasil ekstraksi dari `hasil_ekstraksi/`
- Generate QA dari hasil ekstraksi yang dimuat

## Alur kerja

1. Jalankan backend Flask.
2. Buka web UI.
3. Masukkan `API key` LandingAI.
4. Seret atau pilih file PDF.
5. Tulis schema JSON dalam textarea.
6. Tekan tombol `RUN` untuk ekstraksi.
7. Hasil ekstraksi disimpan di `results/`.
8. Gunakan panel kategori untuk memuat hasil ekstraksi yang sudah ada.
9. Klik tombol `Generate QA` untuk menghasilkan QA dari file yang dimuat.

## Notes

- Hasil ekstraksi baru dapat diunduh sebagai JSON.
- Hasil ekstraksi kategori yang sudah ada dibaca dari folder `hasil_ekstraksi/`.
- `landingai_extractor.html` juga mendukung batch upload serta download QA.
