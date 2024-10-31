import streamlit as st
import pandas as pd
import os
from transformers import pipeline
from dotenv import load_dotenv
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import speech_recognition as sr

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize models
summarizer = pipeline("summarization")

# Streamlit page setup
st.set_page_config(page_title="Multi-Feature Summarizer", page_icon="📝", layout="wide")
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a feature", ["CSV Summarizer", "YouTube Summarizer", "Text Summarizer", "Speech-to-Text Summarizer"])

# CSV Summarizer
if app_mode == "CSV Summarizer":
    st.title("📰 CSV News Article Summarizer")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(df)

        if 'text' in df.columns:
            def summarize_article(article):
                max_chunk_size = 1024
                chunks = [article[i:i+max_chunk_size] for i in range(0, len(article), max_chunk_size)]
                summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
                return " ".join(summaries)
            
            df['Summary'] = df['text'].apply(lambda x: summarize_article(x) if pd.notna(x) else "Not enough content")
            st.write("Summarized Articles:")
            st.dataframe(df[['headline', 'Summary']])
            
            csv = df.to_csv().encode('utf-8')
            st.download_button("Download summarized data as CSV", data=csv, file_name='summarized_articles.csv', mime='text/csv')
        else:
            st.error("CSV file must contain a 'text' column.")

# YouTube Summarizer
elif app_mode == "YouTube Summarizer":
    st.title("🎥 YouTube Video Summarizer")
    youtube_link = st.text_input("Enter YouTube Video Link:")
    
    if youtube_link:
        video_id = youtube_link.split("=")[1]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

    def extract_transcript(video_id):
        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = " ".join([i['text'] for i in transcript_data])
            return transcript
        except Exception as e:
            st.error(f"Error retrieving transcript: {e}")
            return None

    if st.button("Get Detailed Notes") and youtube_link:
        transcript_text = extract_transcript(video_id)
        
        if transcript_text:
            prompt = """
                You are a YouTube video summarizer. You will be given a transcript text,
                and you need to summarize the entire video. Provide the summary in bullet points,
                and ensure that the total length does not exceed **1000 words**.
                Only highlight the main points in a concise manner.
            """
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt + transcript_text)
            st.markdown("## Detailed Notes:")
            st.write(response.text)

# Text Summarizer
elif app_mode == "Text Summarizer":
    st.title("📝 Smart Text Summarizer")
    min_words = st.sidebar.slider("Minimum Summary Length (words)", min_value=10, max_value=100, value=25, step=5)
    max_words = st.sidebar.slider("Maximum Summary Length (words)", min_value=50, max_value=300, value=100, step=10)

    input_text = st.text_area("Enter the text you want to summarize", height=250)
    if st.button("Generate Summary"):
        if input_text.strip():
            summary = summarizer(input_text, max_length=max_words, min_length=min_words, do_sample=False)
            summarized_text = summary[0]["summary_text"]
            st.write("### Here is your summary:")
            st.success(summarized_text)
            st.write(f"**Summary Length:** {len(summarized_text.split())} words")
        else:
            st.error("Please provide some text to summarize!")

# Speech-to-Text Summarizer
elif app_mode == "Speech-to-Text Summarizer":
    st.title("🎙️ Speech-to-Text Summarizer")
    min_words = st.sidebar.slider("Minimum Summary Length (words)", min_value=10, max_value=100, value=25, step=5)
    max_words = st.sidebar.slider("Maximum Summary Length (words)", min_value=50, max_value=300, value=100, step=10)

    recognizer = sr.Recognizer()

    def transcribe_audio():
        with sr.Microphone() as source:
            st.write("Please speak now...")
            audio_data = recognizer.listen(source)
            try:
                return recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                st.error("Could not understand the audio.")
            except sr.RequestError:
                st.error("Could not request results from the service.")

    if st.button("Speak to Summarize"):
        spoken_text = transcribe_audio()
        if spoken_text:
            st.success(f"You said: {spoken_text}")
            summary = summarizer(spoken_text, max_length=max_words, min_length=min_words, do_sample=False)
            st.write("### Summary:")
            st.success(summary[0]["summary_text"])
        else:
            st.error("Could not transcribe audio.")
