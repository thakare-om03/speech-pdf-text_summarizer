import streamlit as st
import PyPDF2
import re
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Initialize the tokenizer and model for Q&A
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Initialize the summarization model
summarizer_tokenizer = T5Tokenizer.from_pretrained("t5-base")
summarizer_model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to clean the extracted text
def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# Function to summarize text
def summarize_text(text):
    preprocess_text = text.strip().replace("\n", " ")
    t5_input_text = "summarize: " + preprocess_text
    tokenized_text = summarizer_tokenizer.encode(t5_input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summarizer_model.generate(tokenized_text, num_beams=4, no_repeat_ngram_size=2, min_length=30, max_length=200, early_stopping=True)
    summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to answer questions based on context
def ask_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Set up the Streamlit app
st.set_page_config(page_title="PDF Summarizer & Q&A Chatbot", page_icon=":book:", layout="centered")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    raw_text = extract_text_from_pdf(uploaded_file)
    cleaned_text = clean_text(raw_text)
    
    # Display extracted text
    st.subheader("Extracted Text")
    st.text_area("Extracted Text", cleaned_text, height=300)
    
    # Summarization section
    if st.button("Summarize"):
        summary = summarize_text(cleaned_text)
        st.subheader("Summary")
        st.success(summary)
    
    # Q&A section
    st.subheader("Ask Questions About the PDF")
    question = st.text_input("Enter your question:")
    if question:
        answer = ask_question(question, cleaned_text)
        st.subheader("Answer")
        st.info(answer)