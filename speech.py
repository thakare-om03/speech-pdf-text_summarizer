import streamlit as st
import os
import google.generativeai as genai
import speech_recognition as sr

# Configure the Google Generative AI Model with the API key
genai.configure(api_key="AIzaSyDjwPzhxqpxgPp7aMHr2k82dy3vMGHj_vc")

# Function to generate a summary using Google Gemini
def generate_gemini_content(transcript_text):
    model = genai.GenerativeModel("gemini-pro")
    # Define a prompt with specific instructions
    prompt = f"""
    You are a speech summarizer. Here is the transcript:
    
    {transcript_text}
    
    Summarize the main points accurately and concisely.
    """
    
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI for Speech to Text Summarizer
st.title("Speech to Text Summarizer")

# Initialize session state to manage recording status
if 'recording' not in st.session_state:
    st.session_state.recording = False

# Function to handle the recording
def start_recording():
    st.session_state.recording = True

# Button to start recording speech
if st.button("Start Listening"):
    start_recording()
    st.write("Listening... Please speak clearly.")
    
    # Initialize the recognizer
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        st.write("Recognizing...")
        audio = recognizer.listen(source)
        
        try:
            # Recognize speech using Google Speech Recognition
            recognized_text = recognizer.recognize_google(audio)
            st.write("Recognized Text:", recognized_text)
            
            # Generate summary if recognized text is available
            summary = generate_gemini_content(recognized_text)
            st.markdown("## Summary:")
            st.write(summary)
        
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Button to start a new recording
if st.button("Start New Recording"):
    # Reset session state for recording
    st.session_state.recording = False
    st.write("Ready to start a new recording.")
    st.stop()  # Stop execution to reset the state
