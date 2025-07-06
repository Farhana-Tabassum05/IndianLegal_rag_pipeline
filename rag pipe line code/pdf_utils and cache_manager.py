import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text


# cache_manager.py
import os
import json

def get_cache_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def is_same_pdf(current_name, cache_file='last_pdf.txt'):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return f.read().strip() == current_name
    return False

def update_last_pdf(current_name, cache_file='last_pdf.txt'):
    with open(cache_file, 'w') as f:
        f.write(current_name)
