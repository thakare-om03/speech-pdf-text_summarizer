import streamlit as st
from transformers import pipeline
import PyPDF2
import re
import pandas as pd
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import speech_recognition as sr
import textwrap

# Page Configuration
st.set_page_config(page_title="Multi-Feature Summarizer", page_icon="📄", layout="wide")

# Initialize pipelines
summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
genai.configure(api_key="YOUR_GOOGLE_API_KEY")

# Helper function to display banner
def display_banner(banner_url, caption=""):
    st.image(banner_url, caption=caption, use_column_width=True)

# Helper function to chunk text for long input handling
def chunk_text(text, max_chunk_length=1024):
    return textwrap.wrap(text, max_chunk_length, break_long_words=False, replace_whitespace=False)

# Header
st.title("📝 Multi-Feature Summarizer")
st.markdown("## Summarize texts, videos, PDFs, and more, all in one place!")

# Tabbed Interface
tabs = st.tabs(["Text Summarization", "Speech Summarization", "PDF Summarizer & Q&A", "YouTube Video Summarizer", "News Article Summarizer"])

# Text Summarization
with tabs[0]:
    display_banner("https://example.com/text_summary_banner.jpg", "Text Summarization")
    st.header("Text Summarization")
    input_text = st.text_area("Enter the text you want to summarize:", height=200)
    st.write(f"Character count: {len(input_text)} / Recommended max: 4096")
    min_words = st.slider("Minimum Summary Length (words)", 10, 100, 25)
    max_words = st.slider("Maximum Summary Length (words)", 50, 300, 100)
    
    if st.button("Generate Summary", key="text_summary_button"):
        if input_text:
            text_chunks = chunk_text(input_text)
            summaries = [summarizer_pipeline(chunk, max_length=max_words, min_length=min_words, do_sample=False)[0]["summary_text"] for chunk in text_chunks]
            full_summary = " ".join(summaries)
            st.subheader("Summary")
            st.success(full_summary)  # Updated to ensure visibility
        else:
            st.error("Please enter text to summarize!")

# Speech Summarization
with tabs[1]:
    display_banner("https://example.com/speech_summary_banner.jpg", "Speech Summarization")
    st.header("Speech Summarization")
    
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if st.button("Start Recording"):
        recognizer = sr.Recognizer()
        st.session_state.recording = True
        with sr.Microphone() as source:
            audio = recognizer.listen(source)
            try:
                transcript = recognizer.recognize_google(audio)
                st.session_state.transcript_text = transcript
                st.success(f"Transcript: {transcript}")
            except sr.UnknownValueError:
                st.error("Could not understand the audio.")
            except sr.RequestError as e:
                st.error(f"Request error: {e}")
    
    if st.session_state.recording and "transcript_text" in st.session_state:
        transcript_text = st.session_state.transcript_text
        summary = summarizer_pipeline(transcript_text, max_length=100, min_length=25, do_sample=False)[0]["summary_text"]
        st.subheader("Summary")
        st.success(summary)  # Updated to ensure visibility

# PDF Summarizer & Q&A
with tabs[2]:
    display_banner("https://example.com/pdf_summary_banner.jpg", "PDF Summarizer & Q&A")
    st.header("PDF Summarizer & Q&A")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        pdf_text = "".join([page.extract_text() for page in pdf_reader.pages])
        st.text_area("Extracted Text", pdf_text, height=200)
        
        if st.button("Summarize PDF"):
            text_chunks = chunk_text(pdf_text)
            summaries = [summarizer_pipeline(chunk, max_length=200, min_length=50, do_sample=False)[0]["summary_text"] for chunk in text_chunks]
            summary = " ".join(summaries)
            st.subheader("Summary")
            st.success(summary)  # Updated to ensure visibility
        
        question = st.text_input("Ask a question about the PDF:")
        if question:
            answer = qa_pipeline(question=question, context=pdf_text)["answer"]
            st.subheader("Answer")
            st.success(answer)  # Updated to ensure visibility

# YouTube Video Summarization
with tabs[3]:
    display_banner("https://example.com/youtube_summary_banner.jpg", "YouTube Video Summarizer")
    st.header("YouTube Video Summarizer")
    
    video_url = st.text_input("Enter YouTube video URL:")
    if video_url:
        video_id = video_url.split("v=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry["text"] for entry in transcript])
        st.text_area("Transcript", transcript_text, height=200)
        
        if st.button("Generate Video Summary"):
            text_chunks = chunk_text(transcript_text)
            summaries = [genai.GenerativeModel("gemini-pro").generate_content("Summarize this video transcript:\n" + chunk).text for chunk in text_chunks]
            full_summary = " ".join(summaries)
            st.subheader("Summary")
            st.success(full_summary)  # Updated to ensure visibility

# News Article Summarizer
with tabs[4]:
    display_banner("https://example.com/news_summary_banner.jpg", "News Article Summarizer")
    st.header("News Article Summarizer")
    
    csv_file = st.file_uploader("Upload a CSV file with news articles", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file)
        st.write("Uploaded Articles:")
        st.dataframe(df)
        
        if 'text' in df.columns:
            summaries = [summarizer_pipeline(text[:1024], max_length=130, min_length=30, do_sample=False)[0]["summary_text"] if pd.notna(text) else "No content" for text in df["text"]]
            df["Summary"] = summaries
            st.write("Summarized Articles:")
            st.dataframe(df[["headline", "Summary"]])
        else:
            st.error("The CSV file must contain a 'text' column.")
