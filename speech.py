import streamlit as st
import google.generativeai as genai
import speech_recognition as sr

# Configure the Google Generative AI Model with the API key
genai.configure(api_key="API_KEY")

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

# Streamlit UI for Speech to Text Summarizer
st.title("Speech to Text Summarizer")

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
                
                # Stop recording after three consecutive timeouts
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
    
# Check if recording has stopped
if not st.session_state.recording and st.session_state.transcript_text.strip():
    summary = generate_gemini_content(st.session_state.transcript_text.strip())
    st.markdown("## Summary:")
    st.write(summary)
