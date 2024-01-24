import fitz  # PyMuPDF for text extraction
import re
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
import numpy as np

def extract_text_from_pdf_with_pymupdf(pdf_path, pages):
    """
    Extract text from specified pages of a PDF using PyMuPDF.

    Args:
        pdf_path (str): The file path to the PDF document.
        pages (list of int): List of page numbers (zero-indexed) to extract text from.

    Returns:
        str: Extracted text from the specified pages.
    """
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num in pages:
            page = doc.load_page(page_num)  # Load the specified page
            text += page.get_text() + "\n"  # Extract text from the page
    return text

def preprocess_text(text):
    """
    Preprocess extracted text by removing non-alphanumeric characters and converting to lowercase.

    Args:
        text (str): The extracted text from the PDF.

    Returns:
        str: Preprocessed text.
    """
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)  # Remove non-alphanumeric characters
    text = text.lower()  # Convert to lowercase
    return text


# Specify the path to the PDF and pages to extract
pdf_path = '/content/ConceptsofBiology-WEB.pdf'  # Path to the PDF file
chapters_to_extract = list(range(18, 68))  # Updated to include pages 19 to 68 (zero-indexed)

extracted_text = extract_text_from_pdf_with_pymupdf(pdf_path, chapters_to_extract)
preprocessed_text = preprocess_text(extracted_text)

# Split the preprocessed text into sentences
sentences = re.split(r'(?<=[.!?]) +', preprocessed_text)

# Load the DistilBERT tokenizer and model for question answering
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')


def find_answer(question, sentences, tokenizer, model):
    max_score = float('-inf')
    best_answer = ""

    for sentence in sentences:
        inputs = tokenizer.encode_plus(question, sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)

        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        input_ids = inputs["input_ids"].tolist()[0]
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        score = torch.max(answer_start_scores) + torch.max(answer_end_scores)
        if score > max_score:
            max_score = score
            best_answer = answer

    return best_answer




# Example usage
example_question = "What are the basic building blocks of molecules?"
answer = find_answer(example_question, sentences, tokenizer, model)
print(answer)
