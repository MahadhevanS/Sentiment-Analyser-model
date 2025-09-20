# --- 1. Import necessary libraries
import streamlit as st
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

@st.cache_resource
def load_resources():
    """Loads the model, stemmer, and stopwords and caches them."""
    try:
        model = joblib.load('sentiment_analyzer.pkl')
        port_stem = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        return model, port_stem, stop_words
    except FileNotFoundError:
        st.error("Error: 'sentiment_analyzer.pkl' not found. Please ensure the model file is in the same directory.")
        return None, None, None

model, port_stem, stop_words = load_resources()

# --- 3. Define the preprocessing function
def stemming(content):
    """Preprocesses a single string for sentiment analysis."""
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [
        port_stem.stem(word) for word in stemmed_content 
        if word not in stop_words
    ]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    layout="centered",
)

st.title("Tweet Sentiment Analyzer")
st.markdown("---")
st.write("Enter a tweet below to analyze its sentiment (positive or negative).")

# Create a text area for user input
user_input = st.text_area("Enter your tweet here:", "")

# Create a button to trigger analysis
if st.button("Analyze Sentiment"):
    if user_input and model:
        with st.spinner('Analyzing...'):
            # Preprocess the user's input using the stemming function
            processed_input = stemming(user_input)
            
            # Make a prediction with the loaded model
            # The model expects a list of inputs, even for a single item
            prediction = model.predict([processed_input])
            
            # Map the numerical prediction to a human-readable label
            sentiment_map = {0: "Negative", 1: "Positive"}
            result = sentiment_map.get(prediction[0], "Unknown")

            # Display the result with a clear message
            if result == "Positive":
                st.success(f"The sentiment is: **{result}**")
            elif result == "Negative":
                st.error(f"The sentiment is: **{result}**")
            else:
                st.warning(f"Could not determine sentiment.")
    elif not user_input:
        st.warning("Please enter some text to analyze.")
    else:
        st.warning("Model could not be loaded. Check your file path.")
