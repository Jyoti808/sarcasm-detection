
import streamlit as st
import tensorflow as tf
import pickle
import json
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Page configuration
st.set_page_config(
    page_title="Sarcasm Detection",
    page_icon="😏",
    layout="centered"
)


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z0-9'\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Load trained model
@st.cache_resource
def load_sarcasm_model():
    return tf.keras.models.load_model(
        "final_sarcasm_lstm_model.keras"
    )


# Load tokenizer
@st.cache_resource
def load_tokenizer():
    with open("sarcasm_tokenizer.pkl", "rb") as file:
        return pickle.load(file)


# Load configuration
with open("sarcasm_model_config.json", "r") as file:
    config = json.load(file)


model = load_sarcasm_model()
tokenizer = load_tokenizer()

max_length = config["max_length"]
threshold = config["classification_threshold"]


# App title
st.title("😏 Sarcasm Detection System")

st.write(
    "Enter a news headline or sentence below to detect "
    "whether it is sarcastic or non-sarcastic."
)


# User input
user_text = st.text_area(
    "Enter Headline:",
    placeholder="Example: Absolutely love waiting three hours at the airport..."
)


# Prediction button
if st.button("Detect Sarcasm"):

    if user_text.strip():

        # Clean input
        cleaned_text = clean_text(user_text)

        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([cleaned_text])

        # Pad sequence
        padded_sequence = pad_sequences(
            sequence,
            maxlen=max_length,
            padding="post",
            truncating="post"
        )

        # Predict
        probability = model.predict(
            padded_sequence,
            verbose=0
        )[0][0]

        # Display prediction
        if probability >= threshold:
            st.error("😏 Prediction: Sarcastic")
        else:
            st.success("🙂 Prediction: Non-Sarcastic")

        st.write(
            f"**Sarcasm Probability:** {probability * 100:.2f}%"
        )

    else:
        st.warning("Please enter a headline first.")
