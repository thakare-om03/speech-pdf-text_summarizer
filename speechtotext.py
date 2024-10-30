import streamlit as st
import speech_recognition as sr
from transformers import pipeline

# Set up page configuration
st.set_page_config(page_title="Smart Summarizer with Speech-to-Text", page_icon="📝", layout="wide")

# Initialize summarization pipeline
summarize_pipeline = pipeline("summarization")

# Function to convert speech to text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Please speak something...")
        audio_data = recognizer.listen(source)
        st.write("Recognizing...")
        try:
            text = recognizer.recognize_google(audio_data)
            st.success("You said: " + text)
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            st.error("Could not request results from the speech recognition service.")
            return None

# Create a sidebar for input and settings
st.sidebar.title("Summarizer Settings ⚙")
st.sidebar.write("Configure your summary settings below:")

# Sidebar elements for user input on minimum and maximum word limits
min_words_input = st.sidebar.slider("Minimum Summary Length (words)", min_value=10, max_value=100, value=25, step=5)
max_words_input = st.sidebar.slider("Maximum Summary Length (words)", min_value=50, max_value=300, value=100, step=10)

# Add a header in the main content area
st.title("📝 Smart Text Summarizer with Speech-to-Text")
st.write("Efficiently summarize long texts using state-of-the-art language models.")

# Button to activate speech-to-text
if st.button("Speak to Summarize"):
    spoken_text = speech_to_text()

# Main input text box
if 'spoken_text' in locals():
    st.write("### Enter the text you want to summarize:")
    input_area = st.text_area("Your input text goes here...", height=250, value=spoken_text if spoken_text else "")
else:
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
        length_placeholder.write(f"*Summary Length:* {len(summarized_text.split())} words")
    else:
        st.error("Please provide some text to summarize!")

# Add an image in the footer for aesthetic appeal
st.sidebar.write("#### Powered by 🤗 Hugging Face Transformers")
st.sidebar.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", use_column_width=True)