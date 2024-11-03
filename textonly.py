import streamlit as st
from transformers import pipeline
import textwrap

# Set up page configuration
st.set_page_config(page_title="Smart Summarizer", page_icon="📝", layout="wide")

# Initialize summarization pipeline with a more robust model
summarize_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Sidebar for summary settings
st.sidebar.title("Summarizer Settings ⚙️")
min_words_input = st.sidebar.slider("Minimum Summary Length (words)", min_value=50, max_value=150, value=75, step=5)
max_words_input = st.sidebar.slider("Maximum Summary Length (words)", min_value=100, max_value=500, value=250, step=10)

# Main page header and instructions
st.title("📝 Smart Text Summarizer")
st.write("Efficiently summarize long texts with optimized settings.")

# Main input text box
st.write("### Enter the text you want to summarize:")
input_area = st.text_area("Your input text goes here...", height=250)
st.write(f"Character count: {len(input_area)} / Recommended max: 4096")

# Summarize button
summarize_button = st.button("Generate Summary")

# Function to chunk large text inputs for summarization
def chunk_text(text, max_chunk_length=1024):
    return textwrap.wrap(text, max_chunk_length, break_long_words=False, replace_whitespace=False)

# Placeholder for displaying summary
summary_placeholder = st.empty()

# Generate and display summary when the button is clicked
if summarize_button:
    if input_area.strip():  # Check if there's input
        # Split text into chunks if too long
        text_chunks = chunk_text(input_area)
        summaries = []

        for chunk in text_chunks:
            # Generate summary for each chunk
            summary_result = summarize_pipeline(chunk, max_length=max_words_input, min_length=min_words_input, 
                                                do_sample=False, no_repeat_ngram_size=2, repetition_penalty=1.5)
            summaries.append(summary_result[0]["summary_text"])

        # Combine chunk summaries and display
        summarized_text = " ".join(summaries)
        summary_placeholder.subheader("Here is your summary:")
        summary_placeholder.success(summarized_text)
        summary_placeholder.write(f"**Summary Length:** {len(summarized_text.split())} words")
    else:
        st.error("Please provide some text to summarize!")

# Footer
st.sidebar.write("#### Powered by 🤗 Hugging Face Transformers")
st.sidebar.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", use_column_width=True)
