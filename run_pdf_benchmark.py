import os
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.document_service import DocumentProcessingService

def create_test_pdf(filename, title, content_paragraphs):
    import subprocess
    import sys
    try:
        from fpdf import FPDF
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf"])
        from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt=title, ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for p in content_paragraphs:
        pdf.multi_cell(0, 10, txt=p)
        pdf.ln(5)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pdf.output(filename)

def run_benchmarks():
    print("=" * 105)
    print(f"{'Modification':<30} | {'Total':<10} | {'Embedded Chunks':<25} | {'Indexing Latency (ms)':<25}")
    print(f"{'Rate (%)':<30} | {'Chunks':<10} | {'Baseline   Proposed':<25} | {'Baseline       Proposed':<25}")
    print("-" * 105)

    scenarios = [
        ("5% (Minor Numerical Edit)", 5, 2),
        ("15% (Paragraph Adjustment)", 15, 6),
        ("30% (Section Revision)", 30, 12),
        ("50% (Multi-Page Restatement)", 50, 20)
    ]

    base_paragraph = "This is a standard paragraph for our synthetic financial report. It contains some generic text that is meant to fill up space so that it reaches approximately four hundred and fifty characters in length. This ensures that when the recursive text splitter processes this document with a chunk size of five hundred, each paragraph will perfectly fit into exactly one single chunk. We will repeat this many times to construct a multi-page document for our robust RAG benchmark tests. "
    
    # Generate 40 paragraphs
    base_paragraphs = [f"{i}. " + base_paragraph for i in range(1, 41)]

    os.makedirs("demo_docs/bench_pdf/v1", exist_ok=True)
    os.makedirs("demo_docs/bench_pdf/v2", exist_ok=True)

    v1_path = "demo_docs/bench_pdf/v1/Financial_Report.pdf"
    create_test_pdf(v1_path, "Q3 Financial Report", base_paragraphs)

    service = DocumentProcessingService()

    for name, rate, num_modified in scenarios:
        v2_paragraphs = list(base_paragraphs)
        # Modify the required number of paragraphs
        for i in range(num_modified):
            v2_paragraphs[i] = v2_paragraphs[i] + " [UPDATED IN V2 FOR BENCHMARK]"

        v2_path = f"demo_docs/bench_pdf/v2/Financial_Report_{rate}pct.pdf"
        create_test_pdf(v2_path, "Q3 Financial Report", v2_paragraphs)

        result = service.benchmark_against_full_rebuild(
            v1_path, 
            v2_path, 
            chunk_size=500, 
            chunk_overlap=50
        )

        base_chunks = result['full_rebuild']['chunks'] # Should be around 40
        baseline_embedded = result['full_rebuild']['embedding_calls']
        proposed_embedded = result['incremental']['embedding_calls']

        baseline_time_ms = result['full_rebuild']['processing_time_seconds'] * 1000
        proposed_time_ms = result['incremental']['processing_time_seconds'] * 1000

        print(f"{name:<30} | {base_chunks:<10} | {baseline_embedded:<10} {proposed_embedded:<14} | {baseline_time_ms:<14.1f} {proposed_time_ms:<10.1f}")

    print("=" * 105)

if __name__ == "__main__":
    run_benchmarks()
