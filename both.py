import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline

# Load environment variables
load_dotenv()

# Function to extract text from PDF files
def extract_pdf_text(pdf_files):
    combined_text = ""
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    combined_text += text
        except Exception as e:
            st.error(f"Error loading PDF: {e}")
    return combined_text

# Split text into manageable chunks
def split_text_into_chunks(text, chunk_size=1200, chunk_overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# Create a vector store for similarity search using SentenceTransformer
def create_vector_store(text_chunks):
    model = SentenceTransformer('hkunlp/instructor-xl')
    embeddings = [model.encode(chunk) for chunk in text_chunks]
    
    # Create FAISS vector store from the embeddings
    vector_store = FAISS.from_embeddings(embeddings)
    return vector_store

# Initialize conversation chain for chat functionality
def initialize_conversation_chain(vector_store):
    language_model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.5)
    memory = ConversationBufferMemory(memory_key='conversation_history', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=language_model,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return chain

# Process user input for chat
def process_user_input(user_input):
    result = st.session_state.conversation({'question': user_input})
    st.session_state.history = result['chat_history']

    for i, message in enumerate(st.session_state.history):
        if i % 2 == 0:
            st.markdown(f"**You:** {message.content}")
        else:
            st.markdown(f"**Bot:** {message.content}")

# Summarization functionality
def generate_summary(input_text, min_len, max_len):
    summarizer = pipeline('summarization')
    summary = summarizer(input_text, min_length=min_len, max_length=max_len, do_sample=False)
    return summary[0]['summary_text']

# Main application function
def main():
    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "history" not in st.session_state:
        st.session_state.history = []

    # Set up page configuration
    st.set_page_config(page_title="PDF Chat & Summarizer", page_icon="📄", layout="wide")

    # Sidebar setup for file uploads
    st.sidebar.title("Document Manager 🗂️")
    pdf_files = st.sidebar.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)
    if st.sidebar.button("Process PDFs"):
        with st.spinner("Processing PDFs..."):
            text_data = extract_pdf_text(pdf_files)
            text_chunks = split_text_into_chunks(text_data)
            vector_store = create_vector_store(text_chunks)
            st.session_state.conversation = initialize_conversation_chain(vector_store)
            st.sidebar.success("Processing completed!")

    # Main tab layout
    st.title("Interactive PDF Chat & Summarization Tool")
    tabs = st.tabs(["💬 Chat with PDFs", "📄 Summarize Text"])

    # Chat tab
    with tabs[0]:
        st.header("Engage in Conversation with your PDFs")
        if st.session_state.conversation:
            user_query = st.text_input("Ask a question based on your uploaded PDFs:")
            if user_query:
                process_user_input(user_query)
        else:
            st.info("Please upload PDFs and process them in the sidebar.")

    # Summarizer tab
    with tabs[1]:
        st.header("Summarize Text Instantly")
        input_text = st.text_area("Enter text to summarize:", height=250)

        # Number input fields for summarization limits
        min_words = st.slider("Minimum summary length (words)", min_value=20, max_value=150, value=40, step=10)
        max_words = st.slider("Maximum summary length (words)", min_value=50, max_value=300, value=120, step=10)

        # Summarize button
        if st.button("Generate Summary"):
            if input_text:
                with st.spinner("Summarizing..."):
                    summary = generate_summary(input_text, min_words, max_words)
                    st.success(f"Summary: {summary}")
                    st.write(f"**Summary Length:** {len(summary.split())} words")
            else:
                st.warning("Please enter text to summarize.")

if __name__ == '__main__':
    main()
