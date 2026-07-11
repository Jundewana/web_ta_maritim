"""
Flask backend for LandingAI Agentic Document Extraction
Run: python app.py
"""

import os
import json
import logging
import requests
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
import re
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.')
CORS(app)

UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'pdf'}
RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Existing results from extraction (kategori-based)
EXTRACTED_RESULTS_DIR = Path(__file__).parent / 'hasil_ekstraksi'

# Ollama Configuration
OLLAMA_ENDPOINT = os.environ.get('OLLAMA_ENDPOINT', 'http://127.0.0.1:11500/api/generate')
PROMPT_DIR = Path(__file__).parent / 'prompt'
PROMPT_TEMPLATES = {
    'default': None,
    'koblex_scenario_atomik': 'prompt_koblex_scenario_atomik.prompt',
    'koblex_scenario': 'prompt_koblex_scenario.prompt',
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    return send_from_directory('.', 'landingai_extractor.html')


@app.route('/extract', methods=['POST'])
def extract():
    """
    Expects multipart/form-data:
      - file: PDF file
      - schema: JSON string
      - api_key: LandingAI API key
    """
    # Validate inputs
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    api_key = request.form.get('api_key', '').strip()
    schema_str = request.form.get('schema', '').strip()

    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file. Only PDF allowed.'}), 400
    if not api_key:
        return jsonify({'error': 'API key is required'}), 400
    if not schema_str:
        return jsonify({'error': 'Schema is required'}), 400

    try:
        schema = json.loads(schema_str)
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Invalid JSON schema: {e}'}), 400

    # Save PDF temporarily
    filename = secure_filename(file.filename)
    pdf_path = Path(UPLOAD_FOLDER) / filename
    file.save(pdf_path)
    logger.info(f"Saved PDF: {pdf_path}")

    try:
        from landingai_ade import LandingAIADE

        client = LandingAIADE(apikey=api_key)

        # Step 1: Parse PDF → Markdown
        logger.info(f"[1/2] Parsing '{filename}'...")
        parse_response = client.parse(document=pdf_path, model="dpt-2")
        markdown = parse_response.markdown
        logger.info(f"[1/2] Parse done. Markdown length: {len(markdown)} chars")

        # Step 2: Extract schema
        logger.info(f"[2/2] Extracting schema from '{filename}'...")
        extract_response = client.extract(
            schema=json.dumps(schema),
            markdown=markdown.encode('utf-8'),
        )
        logger.info(f"[2/2] Extraction done for '{filename}'")

        # Normalize result to plain dict
        result = _to_dict(extract_response)

        # Persist extraction result so it can be previewed later
        try:
            out_name = f"{Path(filename).stem}_extracted.json"
            out_path = RESULTS_DIR / out_name
            with out_path.open('w', encoding='utf-8') as fh:
                json.dump(result, fh, ensure_ascii=False, indent=2)
            logger.info(f"Saved extraction result: {out_path}")
        except Exception as se:
            logger.warning(f"Failed to persist result for '{filename}': {se}")

        return jsonify({
            'status': 'ok',
            'filename': filename,
            'result': result
        })

    except Exception as e:
        logger.error(f"Extraction failed for '{filename}': {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        # Cleanup temp file
        try:
            pdf_path.unlink()
        except Exception:
            pass


def _to_dict(obj):
    """Convert LandingAI response object to plain dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, 'extraction'):
        val = obj.extraction
        return val if isinstance(val, dict) else (vars(val) if hasattr(val, '__dict__') else val)
    if hasattr(obj, '__dict__'):
        return {k: v for k, v in vars(obj).items() if not k.startswith('_')}
    return obj


def _enforce_json_output(prompt):
    suffix = """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PENTING: OUTPUT HANYA JSON VALID
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Keluaran HARUS berupa SATU objek JSON valid saja.
- Top-level HARUS berisi field `qa_pairs`.
- `qa_pairs` HARUS berupa list QA objects.
- Jangan sertakan markdown, kode fences, atau teks tambahan apa pun di luar JSON.
- Jika Anda menggunakan beberapa objek JSON di luar `qa_pairs`, hasil dianggap tidak valid.
- Untuk prompt KOBLEX, `qa_pairs` harus berisi minimal 4 item.
"""
    return prompt + suffix


def _load_prompt_template(prompt_choice, extracted_data):
    if prompt_choice not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt choice: {prompt_choice}")
    template_filename = PROMPT_TEMPLATES[prompt_choice]
    if template_filename is None:
        return _default_qa_prompt(extracted_data)

    prompt_path = PROMPT_DIR / template_filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    source = prompt_path.read_text(encoding='utf-8')
    scope = {
        'json': json,
        'extracted_data': extracted_data,
    }
    exec(source, scope)
    prompt = scope.get('prompt')
    if not isinstance(prompt, str):
        raise ValueError(f"Prompt template {template_filename} did not define a prompt string")
    return _enforce_json_output(prompt)


def _default_qa_prompt(extracted_data):
    return f"""Anda adalah generator dataset QA multi-hop untuk sistem RAG hukum perikanan dan pelayaran Indonesia.
Tugas Anda menghasilkan pasangan pertanyaan-jawaban yang terasa seperti pertanyaan nyata dari pengguna awam kepada chatbot hukum — bukan pertanyaan akademik atau ujian. Ikuti semua aturan di bawah ini dengan ketat.

PENTING: Jangan menyensor atau mengubah nama asli dari dokumen (misal: nama terdakwa, nama kapal, lokasi). Gunakan nama asli secara utuh.

Berdasarkan ringkasan hasil ekstraksi dokumen putusan berikut (dalam format JSON):

{json.dumps(extracted_data, indent=2, ensure_ascii=False)}

- R1_natural:
Pertanyaan HARUS terasa seperti input nyata pengguna ke chatbot, bukan kalimat akademik. Gunakan bahasa percakapan, kalimat tidak sempurna, atau framing personal ('Saya punya teman...', 'Misalnya kalau...'). Hindari kalimat formal seperti 'Dalam konteks hukum pidana Indonesia, jelaskan...'
- R2_regulasi:
JIKA pertanyaan menyebut atau mengandalkan regulasi tertentu, WAJIB cantumkan nomor dokumen hukum yang spesifik.
Contoh benar: 'Berdasarkan Pasal 3 huruf b Permen KP No. 56/PERMEN-KP/2016, apakah...'
Contoh salah: 'Berdasarkan peraturan menteri terkait kepiting bertelur, apakah...'
- R3_tanpa_regulasi:
JIKA pertanyaan tidak menyebut regulasi secara spesifik, gunakan salah satu dari tiga gaya:
(a) SKENARIO: 'Misalkan seorang pengepul ikan di Balikpapan ketahuan menyimpan kepiting bertelur di cold storage-nya...'
(b) ANALOGI: 'Sama seperti SIM yang tidak membebaskan pengemudi dari pidana tabrak lari, apakah izin usaha perikanan...'
(c) BERBASIS CONTOH: 'Dalam kasus Miekie Hidayat yang diadili di PN Jakarta Utara tahun 2019...'
- R4_self_explained:
Pertanyaan HARUS dapat dipahami tanpa membaca dokumen sumber. Jika ada konteks yang diperlukan, masukkan langsung ke dalam kalimat pertanyaan — jangan simpan di field 'konteks_skenario' saja.
- R5_multihop:
Pertanyaan harus memerlukan minimal 2 hop penalaran: menghubungkan fakta spesifik + norma hukum, atau membandingkan dua pasal, atau menganalisis implikasi dari rangkaian kejadian.
- R6_no_leading:
Jangan bocorkan jawaban di dalam pertanyaan. Pertanyaan yang diakhiri dengan 'Apakah benar bahwa...' dan lalu menyebutkan jawabannya adalah pertanyaan yang buruk.

ATURAN JAWABAN:
- R7_fakta_spesifik:
Jawaban WAJIB menyebut fakta konkret dari dokumen putusan: nama pihak, nomor HC, nomor kontainer, tanggal, nominal denda, nama hakim — sesuai yang relevan.
- R8_pasal_lengkap:
Setiap referensi hukum harus menyebut nomor pasal lengkap, nama undang-undang, dan tahun.
Contoh benar:
'Pasal 88 jo. Pasal 16 ayat (1) UU No. 31 Tahun 2004 tentang Perikanan sebagaimana diubah UU No. 45 Tahun 2009'
Contoh salah:
'Pasal 88 UU Perikanan'
- R9_struktur:
Jawaban harus terstruktur:
(1) jawaban langsung atas pertanyaan,
(2) dasar hukum atau fakta pendukung,
(3) nuansa atau pengecualian jika ada.
- R10_panjang:
Jawaban minimal 120 kata, maksimal 350 kata. Cukup komprehensif untuk berguna, cukup ringkas untuk dibaca di chatbot.

ATURAN GRAPH RELATIONS:
- R11_tipe_predikat:
Setiap triplet wajib memiliki field 'predicate_type' dengan nilai:
- 'faktual' (langsung dari dokumen)
- 'normatif' (hubungan antar norma hukum)
- 'prosedural' (urutan atau kewenangan)
- 'inferensial' (kesimpulan dari penalaran multi-hop)
- R12_inferensial_wajib:
Setiap QA pair HARUS mengandung minimal 1 triplet bertipe 'inferensial' — ini yang membedakan multi-hop dari single-lookup.
- R13_jumlah:
Minimal 4 triplet, maksimal 8 triplet per QA pair.
- R14_subjek_spesifik:
Subjek dan objek triplet harus spesifik, bukan generik.
'Pasal 88 UU No. 31 Tahun 2004' lebih baik dari 'UU Perikanan'.
'Miekie Hidayat' lebih baik dari 'Terdakwa'.

CONTOH PERTANYAAN BAIK:
- "Saya baca Permen KP No. 56/PERMEN-KP/2016 — di situ dilarang 'mengumpulkan' kepiting bertelur. Tapi saya nelayan kecil yang kadang dapat kepiting bertelur nyasar ke jaring saya. Apakah saya otomatis melanggar kalau saya tidak langsung buang ke laut?"
- "Dalam kasus Miekie Hidayat tahun 2019, kenapa health certificate dari BKIPM bisa terbit padahal isi kontainernya kepiting bertelur semua? Apa yang salah dari sistem pengawasannya?"
- "Kalau perusahaan saya punya IUP lengkap, HACCP, semua izin beres — lalu ketahuan saya ikut ekspor kepiting bertelur tanpa sengaja karena salah sortir di gudang — apakah izin-izin itu bisa jadi alasan pembenar di pengadilan?"

CONTOH PERTANYAAN BURUK:
- "Dalam konteks hukum pidana Indonesia, jelaskan hubungan antara Permen KP dan UU Perikanan dalam kasus kepiting bertelur."
[SALAH: terlalu akademik, bukan gaya user chatbot]
- "Berdasarkan peraturan yang berlaku, apakah tindakan terdakwa memenuhi unsur deelneming?"
[SALAH: menyebut konsep hukum tanpa konteks, tidak self-explained]
- "Apakah benar bahwa denda Rp100 juta dalam putusan kasus ini tidak proporsional karena nilainya hanya 3-7% dari nilai barang?"
[SALAH: pertanyaan bocorkan jawabannya sendiri]

TIPE TIER (pilih salah satu yang paling cocok dengan pertanyaan/masalah utama):
cross_pasal, timeline_multi_aktor, norma_kontradiksi, hipotetis, analogis, proporsionalitas


OUTPUT FORMAT WAJIB:
Hasilkan output HANYA dalam bentuk JSON valid dengan list "qa_pairs" yang berisi SATU item dengan skema berikut (jangan tambahkan teks markdown ```json di sekitarnya jika bisa dihindari, pastikan murni array/object):
{{
  "id": "uuid atau string unik",
  "tier": "tipe_tier_di_atas",
  "persona_penanya": "deskripsi singkat (misal: 'Nelayan kecil', 'Pemilik kapal')",
  "user_input_style": "skenario / analogi / dll",
  "question": "pertanyaan kompleks",
  "answer": "jawaban analisis hukum sesuai R7-R10",
  "reasoning_steps": ["langkah 1", "langkah 2", "langkah 3"],
  "referensi_hukum": ["pasal 1", "UU 2"],
  "graph_relations": [
     {{"subject": "A", "predicate": "melakukan", "object": "B", "predicate_type": "faktual"}}
  ],
  "konteks_tambahan": "opsional"
}}"""

# ============================================================
# RESULTS API
# ============================================================
@app.route('/results/list')
def results_list():
    items = []
    for p in sorted(RESULTS_DIR.glob('*_extracted.json'), key=lambda x: x.stat().st_mtime, reverse=True):
        stat = p.stat()
        items.append({
            'name': p.name,
            'filename': p.name.replace('_extracted.json', ''),
            'mtime': stat.st_mtime,
            'size': stat.st_size,
        })
    return jsonify(items)


@app.route('/results/file/<path:name>')
def results_file(name):
    safe = secure_filename(name)
    p = RESULTS_DIR / safe
    if not p.exists():
        return jsonify({'error': 'not found'}), 404
    return send_from_directory(str(RESULTS_DIR), safe)


# ============================================================
# EXTRACTED RESULTS BY CATEGORY (hasil_ekstraksi)
# ============================================================
@app.route('/results/categories')
def results_categories():
    """List available categories (pelayaran, perikanan, etc.)"""
    categories = []
    if EXTRACTED_RESULTS_DIR.exists():
        for cat_dir in sorted(EXTRACTED_RESULTS_DIR.iterdir()):
            if cat_dir.is_dir():
                # Count JSON files in category
                count = len(list(cat_dir.glob('*_extracted.json')))
                if count > 0:
                    categories.append({
                        'name': cat_dir.name,
                        'count': count
                    })
    return jsonify(categories)


@app.route('/results/by-category/<category>')
def results_by_category(category):
    """List extraction results by category"""
    cat_path = EXTRACTED_RESULTS_DIR / secure_filename(category)
    if not cat_path.exists() or not cat_path.is_dir():
        return jsonify({'error': 'category not found'}), 404
    
    items = []
    for p in sorted(cat_path.glob('*_extracted.json'), key=lambda x: x.stat().st_mtime, reverse=True):
        stat = p.stat()
        items.append({
            'name': p.name,
            'filename': p.name.replace('_extracted.json', ''),
            'category': category,
            'mtime': stat.st_mtime,
            'size': stat.st_size,
        })
    return jsonify(items)


@app.route('/results/file-by-category/<category>/<path:name>')
def results_file_by_category(category, name):
    """Get file from specific category"""
    cat_path = EXTRACTED_RESULTS_DIR / secure_filename(category)
    if not cat_path.exists() or not cat_path.is_dir():
        return jsonify({'error': 'category not found'}), 404
    
    # Don't sanitize 'name' again - it's from actual filesystem
    # Just validate it exists and is a file
    p = cat_path / name
    if not p.exists() or not p.is_file():
        return jsonify({'error': 'not found'}), 404
    
    return send_from_directory(str(cat_path), name)


def _generate_qa_from_extraction(extracted_data, prompt_choice='default', model="gemma3:12b"):
    """
    Generate QA from extracted document using Ollama or OpenAI.
    Returns dict containing the GraphRAG fields based on multihop-qa-graphrag2.json schema.
    """
    prompt = _load_prompt_template(prompt_choice, extracted_data)

    try:
        if model == "openai/chatgpt":
            logger.info("Calling OpenAI API")
            openai_api_key = os.environ.get('OPENAI_API_KEY')
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set in environment or .env file.")
            
            headers = {
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o",
                "response_format": { "type": "json_object" },
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            data = response.json()
            result_text = data['choices'][0]['message']['content'].strip()
        else:
            logger.info(f"Calling Ollama endpoint: {OLLAMA_ENDPOINT} with model {model}")
            response = requests.post(
                OLLAMA_ENDPOINT,
                json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
                timeout=300
            )
            response.raise_for_status()
            result_text = response.json().get('response', '').strip()
        
        if not result_text:
            raise ValueError(f"Empty response from {model}")
        
        logger.info(f"{model} response received ({len(result_text)} chars)")
        preview = result_text[:1200].replace('\n', '\\n')
        logger.info(f"RAW RESULT TEXT PREVIEW: {preview}")
        
        # Parse structured output
        parsed = _parse_qa_response(result_text)
        parsed = _normalize_parsed_qa(parsed)
        if isinstance(parsed, dict) and isinstance(parsed.get('qa_pairs'), list):
            logger.info(f"PARSED QA count: {len(parsed['qa_pairs'])}")
            if prompt_choice in ('koblex_scenario_atomik', 'koblex_scenario') and len(parsed['qa_pairs']) < 4:
                raise ValueError(f"LLM response contained fewer than 4 QA pairs ({len(parsed['qa_pairs'])}) for prompt {prompt_choice}")
        logger.info(f"PARSED RESULT: {json.dumps(parsed, ensure_ascii=False)}")
        return parsed
        
    except requests.exceptions.ConnectionError:
        raise Exception(f"Cannot connect to endpoint. If using Ollama, make sure it is running at {OLLAMA_ENDPOINT}.")
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise


def _is_qa_like_object(obj):
    if not isinstance(obj, dict):
        return False
    keys = set(obj.keys())
    if keys & {'question', 'answer', 'pertanyaan', 'jawaban', 'background_scenario', 'reasoning_steps'}:
        return True
    if 'id' in obj and ('tier' in obj or 'user_input_style' in obj or 'reference_provisions' in obj):
        return True
    return False


def _find_nested_qa_payload(obj):
    if isinstance(obj, dict):
        if 'qa_pairs' in obj and isinstance(obj['qa_pairs'], list):
            return obj
        for key, value in obj.items():
            if isinstance(value, dict):
                nested = _find_nested_qa_payload(value)
                if nested is not None:
                    return nested
            elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                if all(_is_qa_like_object(item) for item in value):
                    return {'qa_pairs': value}
    return None


def _normalize_parsed_qa(parsed):
    if isinstance(parsed, dict) and 'qa_pairs' not in parsed and _is_qa_like_object(parsed):
        return {'qa_pairs': [parsed]}
    return parsed


def _wrap_qa_object(data):
    if isinstance(data, dict):
        nested = _find_nested_qa_payload(data)
        if nested is not None:
            return nested
        if 'qa' in data:
            if isinstance(data['qa'], list) and all(isinstance(item, dict) for item in data['qa']):
                return {'qa_pairs': data['qa']}
            if isinstance(data['qa'], dict):
                nested = _find_nested_qa_payload(data['qa'])
                if nested is not None:
                    return nested
        if _is_qa_like_object(data):
            return {'qa_pairs': [data]}
    return data


def _parse_qa_response(text):
    """
    Parse structured QA JSON response.
    """
    text = text.strip()
    if not text:
        return {'error': 'Empty response'}

    # Remove common markdown code fences around the JSON response
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    # Try parsing the whole text first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return {'qa_pairs': data}
        return _wrap_qa_object(data)
    except json.JSONDecodeError:
        pass

    # Try to locate the first JSON object/array in the text
    first_json = re.search(r'[\[{]', text)
    if first_json:
        text = text[first_json.start():]

    decoder = json.JSONDecoder()
    idx = 0
    parsed = []
    while idx < len(text):
        try:
            obj, end = decoder.raw_decode(text, idx)
            parsed.append(obj)
            idx = end
            while idx < len(text) and text[idx].isspace():
                idx += 1
        except json.JSONDecodeError:
            break

    if len(parsed) == 1:
        single = parsed[0]
        if isinstance(single, list):
            return {'qa_pairs': single}
        return _wrap_qa_object(single)
    if len(parsed) > 1:
        return {'qa_pairs': parsed}

    logger.error(f"Failed to parse JSON QA: no valid JSON found | RAW TEXT: {text}")
    return {
        'error': 'Failed to parse JSON QA',
        'raw_text': text
    }


@app.route('/generate-qa', methods=['POST'])
def generate_qa():
    """
    Generate QA from extracted document.
    Expects JSON body with 'extracted_data' field.
    """
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({'error': 'No JSON payload provided'}), 400
        
        extracted_data = payload.get('extracted_data')
        model = payload.get('model', 'gemma3:12b')
        prompt_choice = payload.get('prompt_choice', 'default')
        if not extracted_data:
            return jsonify({'error': 'extracted_data is required'}), 400
        
        logger.info(f"Starting QA generation with model: {model} and prompt choice: {prompt_choice}...")
        qa_result = _generate_qa_from_extraction(extracted_data, prompt_choice=prompt_choice, model=model)
        
        logger.info("QA generation completed")
        return jsonify({
            'status': 'ok',
            'qa': qa_result
        })
        
    except Exception as e:
        logger.error(f"QA generation failed: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'upload_dir': UPLOAD_FOLDER})


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  ADE Backend — LandingAI Extractor")
    print("  http://localhost:5123")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5123, debug=False)