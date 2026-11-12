import pytest
import time
import os
import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.document_service import DocumentProcessingService

class StandardRAGService(DocumentProcessingService):
    """
    Simulates a standard RAG pipeline that re-embeds all chunks 
    regardless of whether they have changed or not.
    """
    def process_document_standard(self, file_path):
        start_time = time.perf_counter()
        filename = Path(file_path).name

        # Extract
        pages = self.loader.extract_text(file_path)

        # Clean
        cleaned_pages = [{"page": p["page"], "text": self.cleaner.clean(p["text"])} for p in pages]

        # Split
        chunks = self.splitter.split_pages(cleaned_pages)

        # Re-embed everything
        embedding_calls = 0
        for chunk in chunks:
            # Simulate embedding
            embedding = self.embedder.embed_text(chunk["text"])
            embedding_calls += 1
            
            chunk_id = f"{filename}_{chunk['chunk_number']}"
            
            # Since this is simulating, we upsert into a dummy/test logic
            # but to actually perform the DB operation we use chroma
            try:
                self.chroma.collection.upsert(
                    ids=[chunk_id],
                    documents=[chunk["text"]],
                    embeddings=[embedding],
                    metadatas=[{
                        "document_name": filename,
                        "page": chunk["page"],
                        "chunk_number": chunk["chunk_number"]
                    }]
                )
            except Exception as e:
                pass # For standard rag simulation, just ignoring chroma errors if any

        end_time = time.perf_counter()
        processing_time = round(end_time - start_time, 3)

        return {
            "filename": filename,
            "chunks": len(chunks),
            "embedding_calls": embedding_calls,
            "processing_time_seconds": processing_time
        }


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

@pytest.fixture(scope="session", autouse=True)
def setup_pdfs():
    """Generate large enough PDFs to ensure multiple chunks are created."""
    # Create 10 large paragraphs
    base_paragraphs = [
        f"{i}. This is a very long paragraph to ensure we exceed the 1000 character limit of the text splitter when combined. " * 10
        for i in range(1, 11)
    ]
    
    para_modified = "2. This is a very long paragraph to ensure we exceed the 1000 character limit of the text splitter when combined. " * 9 + "MODIFIED TEXT."
    
    v1_paragraphs = list(base_paragraphs)
    v2_paragraphs = list(base_paragraphs)
    v2_paragraphs[1] = para_modified # Modify the second paragraph
    
    os.makedirs("demo_docs/v1", exist_ok=True)
    os.makedirs("demo_docs/v2", exist_ok=True)
    
    create_test_pdf("demo_docs/v1/Acme_Report.pdf", "Acme Corp Financial Report", v1_paragraphs)
    create_test_pdf("demo_docs/v2/Acme_Report.pdf", "Acme Corp Financial Report", v2_paragraphs)


def test_benchmark_incremental_vs_standard():
    print("\n--- Starting Benchmark: Incremental vs Standard RAG ---\n")
    
    # Original paths
    v1_path_orig = "demo_docs/v1/Acme_Report.pdf"
    v2_path_orig = "demo_docs/v2/Acme_Report.pdf"
    
    # Test paths so we don't collide with existing 'Acme_Report.pdf' in manifest
    # or between standard and incremental
    os.makedirs("demo_docs/test", exist_ok=True)
    
    std_test_pdf = "demo_docs/test/std_Acme_Report.pdf"
    inc_test_pdf = "demo_docs/test/inc_Acme_Report.pdf"
    
    # --- 1. Run Standard RAG benchmark ---
    standard_service = StandardRAGService()
    
    # Copy v1 to standard test path
    shutil.copy(v1_path_orig, std_test_pdf)
    print(f"Standard RAG: Processing v1 (File: {Path(std_test_pdf).name})")
    std_v1_results = standard_service.process_document_standard(std_test_pdf)
    
    # Copy v2 to standard test path
    shutil.copy(v2_path_orig, std_test_pdf)
    print("Standard RAG: Processing v2...")
    std_v2_results = standard_service.process_document_standard(std_test_pdf)
    
    
    # --- 2. Run Incremental RAG benchmark ---
    incremental_service = DocumentProcessingService()
    
    # Copy v1 to incremental test path
    shutil.copy(v1_path_orig, inc_test_pdf)
    print(f"\nIncremental RAG: Processing v1 (File: {Path(inc_test_pdf).name})")
    inc_v1_results = incremental_service.process_document(inc_test_pdf)
    
    # Copy v2 to incremental test path
    shutil.copy(v2_path_orig, inc_test_pdf)
    print("Incremental RAG: Processing v2...")
    inc_v2_results = incremental_service.process_document(inc_test_pdf)
    
    # --- Compare Results ---
    print(f"\n=============================================")
    print(f"Results Comparison for Updating (v1 -> v2):")
    print(f"=============================================")
    print(f"Standard Embedding Calls: {std_v2_results['embedding_calls']}")
    print(f"Incremental Embedding Calls: {inc_v2_results['embedding_calls']}")
    print(f"Standard Time: {std_v2_results['processing_time_seconds']}s")
    print(f"Incremental Time: {inc_v2_results['processing_time_seconds']}s")
    print(f"=============================================\n")
    
    assert inc_v2_results['embedding_calls'] < std_v2_results['embedding_calls'], \
        "Incremental RAG should use fewer embedding calls than Standard RAG when updating."
    
    # Cleanup
    if os.path.exists(std_test_pdf):
        os.remove(std_test_pdf)
    if os.path.exists(inc_test_pdf):
        os.remove(inc_test_pdf)


def test_benchmark_can_run_with_custom_storage_paths(tmp_path):
    custom_manifest = tmp_path / "benchmark_manifest.db"
    service = DocumentProcessingService(
        manifest_db_path=str(custom_manifest),
        chroma_collection_name="benchmark_collection"
    )

    assert service.sqlite.db_path == str(custom_manifest)
    assert service.chroma.collection.name == "benchmark_collection"
