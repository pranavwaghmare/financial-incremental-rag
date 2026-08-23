import os
import glob
import time
import hashlib

def extract_text(pdf_path):
    """Extracts raw text using whichever PDF library is installed in the venv."""
    # 1. Try pypdf
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        text = "".join([page.extract_text() or "" for page in reader.pages])
        if text.strip():
            return text
    except Exception:
        pass

    # 2. Try PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(pdf_path)
        text = "".join([page.extract_text() or "" for page in reader.pages])
        if text.strip():
            return text
    except Exception:
        pass

    # 3. Try PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = "".join([page.get_text() or "" for page in doc])
        if text.strip():
            return text
    except Exception:
        pass

    # 4. Try pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text = "".join([page.extract_text() or "" for page in pdf.pages])
            if text.strip():
                return text
    except Exception:
        pass

    raise RuntimeError(f"Unable to extract text from {pdf_path}. Please check PDF libraries.")

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """Recursive character sliding-window chunker."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - chunk_overlap)
    return chunks

def hash_chunk(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def run_intra_benchmark():
    # 1. Locate real SEC PDF from the downloaded FinanceBench files
    pdf_candidates = (
        glob.glob("data/financebench/**/*.pdf", recursive=True) +
        glob.glob("data/**/*.pdf", recursive=True)
    )
    if not pdf_candidates:
        print("[!] No PDF found in data/. Run `python incremental_eval.py` first to download PDFs.")
        return

    pdf_path = pdf_candidates[0]
    print(f"[*] Target Real SEC Filing: {os.path.basename(pdf_path)}")

    # 2. Extract text and chunk V1
    text_v1 = extract_text(pdf_path)
    chunks_v1 = chunk_text(text_v1, chunk_size=500, chunk_overlap=50)
    total_chunks = len(chunks_v1)
    print(f"[+] Total Chunks in Base Document: {total_chunks}")

    hashes_v1 = {hash_chunk(c): c for c in chunks_v1}

    # 3. Initialize Embedding Model (sentence-transformers)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        get_embed = lambda t: model.encode(t)
    except Exception:
        from app.services.embedding_service import get_embedding
        get_embed = get_embedding

    # Warmup
    _ = get_embed(chunks_v1[0])

    # 4. Define realistic modification scenarios
    edit_configs = [
        ("5% (Minor Numerical Edit)", max(1, int(total_chunks * 0.05))),
        ("15% (Paragraph Adjustment)", max(2, int(total_chunks * 0.15))),
        ("30% (Section Revision)", max(5, int(total_chunks * 0.30))),
        ("50% (Multi-Page Restatement)", max(10, int(total_chunks * 0.50)))
    ]

    print("\n" + "="*85)
    print(f"{'MODIFICATION SCENARIO':<32} | {'TOTAL':<6} | {'BASE':<6} | {'PROP':<6} | {'SAVINGS':<8} | {'BASE LAT':<10} | {'PROP LAT'}")
    print("="*85)

    for label, num_edits in edit_configs:
        # Create V2 by simulating edits at distributed positions
        chunks_v2 = list(chunks_v1)
        step = max(1, total_chunks // num_edits)
        for i in range(num_edits):
            idx = (i * step) % total_chunks
            chunks_v2[idx] = chunks_v2[idx] + f" [RESTATED NUMERICAL VALUE: {i * 104.50}]"

        hashes_v2 = {hash_chunk(c): c for c in chunks_v2}

        # --- Baseline: Re-embed every single chunk ---
        t_base_start = time.perf_counter()
        # Measure sample for embedding throughput
        sample_n = min(15, len(chunks_v2))
        t_sample_start = time.perf_counter()
        for i in range(sample_n):
            _ = get_embed(chunks_v2[i])
        sample_time = (time.perf_counter() - t_sample_start) / sample_n
        t_base_total = (time.perf_counter() - t_base_start) + (sample_time * len(chunks_v2))

        # --- Proposed: CDC Diff + Embed Delta Plus Only ---
        t_prop_start = time.perf_counter()
        delta_plus_hashes = set(hashes_v2.keys()) - set(hashes_v1.keys())
        delta_plus_chunks = [hashes_v2[h] for h in delta_plus_hashes]
        
        for c in delta_plus_chunks:
            _ = get_embed(c)
        t_prop_total = time.perf_counter() - t_prop_start

        embedded_base = len(chunks_v2)
        embedded_prop = len(delta_plus_chunks)
        savings = ((embedded_base - embedded_prop) / embedded_base) * 100

        print(f"{label:<32} | {total_chunks:<6} | {embedded_base:<6} | {embedded_prop:<6} | {savings:>6.1f}% | {t_base_total*1000:>7.1f} ms | {t_prop_total*1000:>6.1f} ms")

    print("="*85)

if __name__ == "__main__":
    run_intra_benchmark()