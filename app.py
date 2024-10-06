import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv()

# Configure the Google Generative AI Model with the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define a prompt with a specific word limit mentioned
prompt = """
You are a YouTube video summarizer. You will be given a transcript text,
and you need to summarize the entire video. Provide the summary in bullet points,
and ensure that the total length does not exceed **1000 words**.
Only highlight the main points in a concise manner.
"""

# Function to extract transcript details from YouTube video URL
def extract_transcript_details(youtube_video_url):
    try:
        # Extract video ID from URL
        video_id = youtube_video_url.split("=")[1]
        
        # Retrieve the transcript using the YouTubeTranscriptApi
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)

        # Combine the transcript into a single string
        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

    except Exception as e:
        raise e

# Function to generate a summary using Google Gemini
def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    # Call the generate_content function without max_output_tokens
    response = model.generate_content(prompt + transcript_text)
    return response.text

# Streamlit UI for YouTube Transcript to Notes Converter
st.title("YouTube lecture to Detailed Notes Converter")

# Input field for YouTube Video Link
youtube_link = st.text_input("Enter YouTube Video Link:")

# Display thumbnail of the YouTube video
if youtube_link:
    video_id = youtube_link.split("=")[1]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

# Button to generate the detailed notes
if st.button("Get Detailed Notes"):
    transcript_text = extract_transcript_details(youtube_link)

    # Generate summary if transcript is available
    if transcript_text:
        # Generate the summary based on the prompt only
        summary = generate_gemini_content(transcript_text, prompt)
        st.markdown("## Detailed Notes:")
        st.write(summary)
