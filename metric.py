import streamlit as st
from transformers import pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Function to calculate evaluation metrics
def calculate_metrics(reference, generated):
    reference_tokens = set(reference.split())
    generated_tokens = set(generated.split())

    true_positives = len(reference_tokens.intersection(generated_tokens))
    false_positives = len(generated_tokens - reference_tokens)
    false_negatives = len(reference_tokens - generated_tokens)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return precision, recall, f1

# Function to plot the metrics
def plot_metrics(precision, recall, f1):
    metrics = ['Precision', 'Recall', 'F1 Score']
    values = [precision, recall, f1]

    fig, ax = plt.subplots()
    ax.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FFC107'])
    ax.set_ylim([0, 1])
    ax.set_ylabel('Scores')
    ax.set_title('Evaluation Metrics')

    return fig

# Set up page configuration
st.set_page_config(page_title="Smart Summarizer", page_icon="📝", layout="wide")

# Specify the model to use
model_name = "sshleifer/distilbart-cnn-12-6"  # Specify the model explicitly
summarize_pipeline = pipeline("summarization", model=model_name)

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

# Input for reference summary
reference_summary = st.text_area("Enter the reference summary for evaluation (optional):", height=100)

# Add a button for summarizing the text
summarize_button = st.button("Generate Summary")

# Placeholders for results
summary_placeholder = st.empty()
length_placeholder = st.empty()
metrics_placeholder = st.empty()
metrics_chart_placeholder = st.empty()

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

        # Calculate and display evaluation metrics if a reference summary is provided
        if reference_summary.strip():
            precision, recall, f1 = calculate_metrics(reference_summary, summarized_text)
            metrics_placeholder.subheader("Evaluation Metrics:")
            metrics_placeholder.write(f"**Precision:** {precision:.2f}")
            metrics_placeholder.write(f"**Recall:** {recall:.2f}")
            metrics_placeholder.write(f"**F1 Score:** {f1:.2f}")

            # Plot and display the metrics as a bar chart
            fig = plot_metrics(precision, recall, f1)
            metrics_chart_placeholder.pyplot(fig)

        else:
            metrics_placeholder.warning("No reference summary provided for evaluation.")
            metrics_chart_placeholder.empty()

    else:
        st.error("Please provide some text to summarize!")

# Add an image in the footer for aesthetic appeal
st.sidebar.write("#### Powered by 🤗 Hugging Face Transformers")
st.sidebar.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", use_column_width=True)
