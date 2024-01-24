import fitz

def extract_text_from_pdf_with_pymupdf(pdf_path, start_page, end_page):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num in range(start_page, end_page + 1):
            page = doc.load_page(page_num)
            text += page.get_text() + "\n"
    return text
