import streamlit as st
import pandas as pd
from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="path_to_fine_tuned_model")

# Streamlit application title
st.title("CNN News Article Summarizer")

# File uploader to upload the CSV dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Display the dataframe
    st.write("Uploaded Data:")
    st.dataframe(df)

    # Check if the required columns are present
    if 'text' in df.columns:
        # Function to handle long articles
        def summarize_long_article(article):
            max_chunk_size = 1024
            article_chunks = [article[i:i+max_chunk_size] for i in range(0, len(article), max_chunk_size)]
            summaries = []
            for chunk in article_chunks:
                summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            return " ".join(summaries)

        # Summarize the articles
        summaries = []
        for article in df['text']:
            if pd.notna(article) and len(article.split()) > 10:  # Check for sufficient content
                try:
                    summary = summarize_long_article(article)
                    summaries.append(summary)
                except Exception as e:
                    summaries.append(f"Error summarizing: {str(e)}")
            else:
                summaries.append("Not enough content to summarize.")

        # Add summaries to the dataframe
        df['Summary'] = summaries
        
        # Display the summaries
        st.write("Summarized Articles:")
        st.dataframe(df[['headline', 'Summary']])
        
        # Download summarized articles as CSV
        def convert_df(df):
            return df.to_csv().encode('utf-8')

        st.download_button(
            label="Download summarized data as CSV",
            data=convert_df(df),
            file_name='summarized_articles.csv',
            mime='text/csv',
        )
    else:
        st.error("CSV file must contain a 'text' column.")
