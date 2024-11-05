import streamlit as st
from transformers import pipeline
import PyPDF2
import re
import pandas as pd
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import speech_recognition as sr
import textwrap

# Initialize Generative AI API
genai.configure(api_key="AIzaSyCbttQBb77_8RGOeDkazcrGWn3v9GjF9hs")

# Initialize pipelines
summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Page Configuration
st.set_page_config(page_title="Multi-Feature Summarizer", page_icon="📄", layout="wide")

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

    # Initialize session state to manage recording status and transcript
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'transcript_text' not in st.session_state:
        st.session_state.transcript_text = ""
    if 'timeout_count' not in st.session_state:
        st.session_state.timeout_count = 0  # Counter for consecutive timeouts

    # Function to handle the recording
    def start_recording():
        st.session_state.recording = True
        st.session_state.transcript_text = ""  # Clear previous transcript
        st.session_state.timeout_count = 0  # Reset timeout count
        st.write("Listening... Please speak clearly.")

        # Initialize the recognizer
        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            st.write("Recognizing...")
            while st.session_state.recording:
                try:
                    audio = recognizer.listen(source, timeout=5)  # Increased timeout for longer pauses
                    recognized_text = recognizer.recognize_google(audio)
                    st.write("Recognized Text:", recognized_text)
                    st.session_state.transcript_text += " " + recognized_text  # Append recognized text to transcript
                    st.session_state.timeout_count = 0  # Reset timeout count on successful recognition

                except sr.UnknownValueError:
                    st.error("Google Speech Recognition could not understand audio.")
                except sr.RequestError as e:
                    st.error(f"Could not request results from Google Speech Recognition service; {e}")
                except sr.WaitTimeoutError:
                    # Handle the case when the listener times out
                    st.session_state.timeout_count += 1
                    st.write("No speech detected for a while. Please continue speaking or stop recording.")

                    # Stop recording after two consecutive timeouts
                    if st.session_state.timeout_count >= 2:
                        st.session_state.recording = False
                        st.write("Recording stopped due to inactivity.")
                        break

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    break

    # Button to start recording speech
    if st.button("Start Recording"):
        start_recording()

    # Button to start recording again
    if st.button("Start Recording Again"):
        st.session_state.recording = False  # Stop any ongoing recording
        st.session_state.transcript_text = ""  # Clear previous transcript
        st.session_state.timeout_count = 0  # Reset timeout count
        st.write("Ready to start a new recording.")

    # Function to generate a summary using Google Gemini
    def generate_gemini_content(transcript_text):
        model = genai.GenerativeModel("gemini-pro")
        prompt = f"""
        You are a speech summarizer. Here is the transcript:
        
        {transcript_text}
        
        Summarize the main points accurately and concisely.
        """
        
        response = model.generate_content(prompt)
        return response.text

    # Check if recording has stopped and transcript is available
    if not st.session_state.recording and st.session_state.transcript_text.strip():
        summary = generate_gemini_content(st.session_state.transcript_text.strip())
        st.markdown("## Summary:")
        st.write(summary)

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
    
    # Input for YouTube URL
    youtube_link = st.text_input("Enter YouTube video URL:")
    
    # Display video thumbnail if URL is entered
    if youtube_link:
        video_id = youtube_link.split("v=")[1]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
    
    # Function to extract transcript
    def extract_transcript_details(youtube_video_url):
        try:
            video_id = youtube_video_url.split("=")[1]
            transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = " ".join([entry["text"] for entry in transcript_text])
            return transcript
        except Exception as e:
            st.error(f"Error retrieving transcript: {e}")
            return None

    # Function to generate summary with Google Gemini
    def generate_gemini_content(transcript_text, prompt):
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + transcript_text)
        return response.text

    # Prompt for the Gemini model
    prompt = """You are a YouTube video summarizer. Take the transcript text
    and summarize the entire video, providing the important points in less than 800 words:
    """

    # Generate video summary if the "Get Detailed Notes" button is clicked
    if st.button("Get Detailed Notes"):
        transcript_text = extract_transcript_details(youtube_link)
        
        if transcript_text:
            summary = generate_gemini_content(transcript_text, prompt)
            st.markdown("## Detailed Notes:")
            st.write(summary)


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
