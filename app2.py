from transformers import pipeline 
import streamlit as st
from PIL import Image 

# Tab name and favicon
st.set_page_config(page_title='Text Summarizer', page_icon='📖', layout='centered')

# Import pipeline for text summarization
summarizer = pipeline('summarization')



# Title and description
st.write("""
# Text Summarizer 💬 
Using Hugging Face Transformers 🤗
""")

# Form layout for user input
with st.form(key='my_form'):
    # Text area for input text
    input_text = st.text_area('Enter your Text', height=300)

    # Use st.columns() instead of the deprecated st.beta_columns()
    left_column, right_column = st.columns(2)

    # User input for min and max words
    min_words = left_column.number_input('Minimum words', value=30)
    max_words = right_column.number_input('Maximum words', value=130)

    # Submit button for the form
    summarize = st.form_submit_button('Summarize!')

# Action triggered on form submission
if summarize:
    summary = summarizer(input_text, max_length=max_words, min_length=min_words, do_sample=False)
    st.subheader('Result 🎉')
    st.info(summary[0]['summary_text'])
    st.write('**Length:** ' + str(len(summary[0]['summary_text'].split(' '))) + ' words')