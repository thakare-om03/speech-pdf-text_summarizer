import streamlit as st
import pandas as pd
from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

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
        # Summarize the articles
        summaries = []
        for article in df['text']:
            # Ensure the article is not empty
            if pd.notna(article) and len(article.split()) > 10:  # Check for sufficient content
                # Check the length of the article
                if len(article) > 1024:  # Truncate if too long
                    article = article[:1024]
                
                try:
                    # Generate a summary for each article
                    summary = summarizer(article, max_length=130, min_length=30, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                except Exception as e:
                    summaries.append(f"Error summarizing: {str(e)}")
            else:
                summaries.append("Not enough content to summarize.")

        # Add summaries to the dataframe
        df['Summary'] = summaries
        
        # Display the summaries
        st.write("Summarized Articles:")
        st.dataframe(df[['headline', 'Summary']])
    else:
        st.error("CSV file must contain a 'text' column.")

# Run the app with: streamlit run app.py
