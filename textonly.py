import streamlit as st
from transformers import pipeline
import textwrap

# Set up page configuration
st.set_page_config(page_title="Smart Summarizer", page_icon="📝", layout="wide")

# Initialize summarization pipeline with a robust model
summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to display a banner image
def display_banner(url, title):
    st.image(url, caption=title, use_column_width=True)

# Function to chunk large text inputs for summarization
def chunk_text(text, max_chunk_length=1024):
    return textwrap.wrap(text, max_chunk_length, break_long_words=False, replace_whitespace=False)

# Tabs for different functionalities
tabs = st.tabs(["Text Summarization", "Other Feature"])

# Text Summarization tab
with tabs[0]:
    display_banner("https://example.com/text_summary_banner.jpg", "Text Summarization")
    st.header("Text Summarization")
    input_text = st.text_area("Enter the text you want to summarize:", height=200)
    st.write(f"Character count: {len(input_text)} / Recommended max: 4096")
    
    # Slider for summary length settings
    min_words = st.slider("Minimum Summary Length (words)", 10, 100, 25)
    max_words = st.slider("Maximum Summary Length (words)", 50, 300, 100)
    
    # Generate Summary button
    if st.button("Generate Summary", key="text_summary_button"):
        if input_text:
            # Split text into chunks if too long
            text_chunks = chunk_text(input_text)
            # Generate summaries for each chunk and combine them
            summaries = [summarizer_pipeline(chunk, max_length=max_words, min_length=min_words, do_sample=False)[0]["summary_text"] for chunk in text_chunks]
            full_summary = " ".join(summaries)
            
            # Display the summary directly
            st.subheader("Summary")
            st.success(full_summary)
        else:
            st.error("Please enter text to summarize!")

# Footer
st.sidebar.write("#### Powered by 🤗 Hugging Face Transformers")
st.sidebar.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", use_column_width=True)