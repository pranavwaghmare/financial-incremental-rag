import os
import subprocess
import sys

# Ensure fpdf is installed
try:
    from fpdf import FPDF
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf"])
    from fpdf import FPDF

def create_pdf(filename, title, content_paragraphs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt=title, ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    for p in content_paragraphs:
        pdf.multi_cell(0, 10, txt=p)
        pdf.ln(5)
        
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pdf.output(filename)

# Base content
para1 = "1. Introduction: This is the financial report for Acme Corp for the year 2025. We have seen significant growth in our core sectors."
para2_v1 = "2. Revenue: Our Q1 revenue was $10 Million, with a profit margin of 15%. This was primarily driven by our software sales."
para2_v2 = "2. Revenue: Our Q1 revenue was $12 Million, with a profit margin of 18%. This was primarily driven by our software sales and new hardware division. (UPDATED)"
para3 = "3. Expenses: Operational expenses were kept under control at $4 Million. We expect this trend to continue."
para4_new = "4. Future Outlook: We are planning to expand into the European market by Q3 2025. (NEW)"

# Version 1
create_pdf(
    "demo_docs/v1/Acme_Report.pdf", 
    "Acme Corp Financial Report", 
    [para1, para2_v1, para3]
)

# Version 2 (para2 modified, para4 added, para3 deleted just to show delete)
create_pdf(
    "demo_docs/v2/Acme_Report.pdf", 
    "Acme Corp Financial Report", 
    [para1, para2_v2, para4_new]
)

print("PDFs generated successfully.")
