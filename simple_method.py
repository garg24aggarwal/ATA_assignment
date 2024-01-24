import fitz  # PyMuPDF for text extraction
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extract_text_from_pdf_with_pymupdf(pdf_path, pages):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num in pages:
            page = doc.load_page(page_num)  # Load the specified page
            text += page.get_text() + "\n"  # Extract text from the page
    return text

def preprocess_text(text):
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

# Initialize TfidfVectorizer and transform the sentences
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

def find_answer(question, sentences, vectorizer, X):
    question_vec = vectorizer.transform([question])  # Vectorize the question
    similarities = cosine_similarity(question_vec, X)  # Calculate cosine similarity with all sentences
    most_similar_idx = np.argmax(similarities)  # Find the index of the most similar sentence
    most_similar_sentence = sentences[most_similar_idx]

    return most_similar_sentence  # Return the most similar sentence

# Example usage
example_question = "What are the basic building blocks of molecules?"
most_similar_sentence = find_answer(example_question, sentences, vectorizer, X)
print(most_similar_sentence)
