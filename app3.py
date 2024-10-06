import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.embeddings import OpenAIEmbeddings
from transformers import pipeline
from PIL import Image

# Utility functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def summarize_text(input_text, min_words, max_words):
    summarizer = pipeline('summarization')
    summary = summarizer(input_text, max_length=max_words, min_length=min_words, do_sample=False)
    return summary[0]['summary_text']

# Main app
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs & Summarize Text", page_icon="📁")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.title("Chat with PDFs & Summarize Text 📁")
    tab1, tab2 = st.tabs(["Chat with PDFs", "Text Summarizer"])

    with tab1:
        st.header("Chat with multiple PDFs :books:")
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)

        with st.sidebar:
            st.subheader("Your documents")
            pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
            if st.button("Process"):
                with st.spinner("Processing"):
                    # Get PDF text
                    raw_text = get_pdf_text(pdf_docs)

                    # Get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)

    with tab2:
        st.header("Text Summarizer 💬")
        input_text = st.text_area('Enter your Text', height=300)

        left_column, right_column = st.columns(2)
        min_words = left_column.number_input('Minimum words', value=30)
        max_words = right_column.number_input('Maximum words', value=130)

        summarize = st.button('Summarize!')

        if summarize:
            summary_text = summarize_text(input_text, min_words, max_words)
            st.subheader('Result 🎉')
            st.info(summary_text)
            st.write('**Length:** ' + str(len(summary_text.split(' '))) + ' words')

if __name__ == '__main__':
    main()