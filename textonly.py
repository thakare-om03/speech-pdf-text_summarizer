import streamlit as st
from transformers import pipeline

# Set up page configuration
st.set_page_config(page_title="Smart Summarizer", page_icon="📝", layout="wide")

# Initialize summarization pipeline
summarize_pipeline = pipeline("summarization")

# Create a sidebar for input and settings
st.sidebar.title("Summarizer Settings ⚙️")
st.sidebar.write("Configure your summary settings below:")

# Sidebar elements for user input on minimum and maximum word limits
min_words_input = st.sidebar.slider("Minimum Summary Length (words)", min_value=10, max_value=100, value=25, step=5)
max_words_input = st.sidebar.slider("Maximum Summary Length (words)", min_value=50, max_value=300, value=100, step=10)

# Add a header in the main content area
st.title("📝 Smart Text Summarizer")
st.write("Efficiently summarize long texts using state-of-the-art language models.")

# Main input text box
st.write("### Enter the text you want to summarize:")
input_area = st.text_area("Your input text goes here...", height=250)

# Add a button for summarizing the text
summarize_button = st.button("Generate Summary")

# Placeholders for results
summary_placeholder = st.empty()
length_placeholder = st.empty()

# Perform summarization if button is clicked
if summarize_button:
    if input_area.strip():  # Check for non-empty input
        # Generate the summary using the provided text
        summary_result = summarize_pipeline(input_area, max_length=max_words_input, min_length=min_words_input, do_sample=False)
        summarized_text = summary_result[0]["summary_text"]

        # Display the summary and its word count
        summary_placeholder.subheader("Here is your summary:")
        summary_placeholder.success(summarized_text)
        length_placeholder.write(f"**Summary Length:** {len(summarized_text.split())} words")
    else:
        st.error("Please provide some text to summarize!")

# Add an image in the footer for aesthetic appeal
st.sidebar.write("#### Powered by 🤗 Hugging Face Transformers")
st.sidebar.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", use_column_width=True)
