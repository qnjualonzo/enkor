import streamlit as st
from transformers import MarianMTModel, MarianTokenizer, BartForConditionalGeneration, BartTokenizer
import re

# Initialize session state variables
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "summarized_text" not in st.session_state:
    st.session_state.summarized_text = ""
if "lang_direction" not in st.session_state:
    st.session_state.lang_direction = "EN to KO"
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Load translation models and tokenizers
translation_models = {
    "EN to KO": {
        "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ko"),
        "tokenizer": MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ko")
    },
    "KO to EN": {
        "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en"),
        "tokenizer": MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
    }
}

# Load summarization model and tokenizer
summarization_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
summarization_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Function to add spaces between sentences
def add_spaces_between_sentences(text):
    return re.sub(r'([.!?])(?=\S)', r'\1 ', text)

# Function to handle translation
def translate_text(input_text, direction):
    try:
        model = translation_models[direction]["model"]
        tokenizer = translation_models[direction]["tokenizer"]
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(**inputs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return ""

# Function to handle summarization
def summarize_text(text, num_sentences=3):
    try:
        inputs = summarization_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = summarization_model.generate(inputs, max_length=130, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        return summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return ""

# Streamlit UI Elements

# Set the title
st.title("EnKoreS: Translation and Summarization")

# Sidebar for language selection
lang_direction = st.sidebar.radio("Select Translation Direction", ["EN to KO", "KO to EN"])

# Reset input text and translations when the direction changes
if lang_direction != st.session_state.lang_direction:
    st.session_state.lang_direction = lang_direction
    st.session_state.input_text = ""
    st.session_state.translated_text = ""
    st.session_state.summarized_text = ""

# Input text area with user instructions
st.subheader("Enter the text you wish to translate:")
st.session_state.input_text = st.text_area(
    "Enter text:", 
    value=st.session_state.input_text, 
    height=200, 
    help="Enter the text that you want to translate and summarize."
)

# Layout for buttons
col1, col2 = st.columns([1, 1])

# Translate button
with col1:
    translate_button = st.button("Translate", key="translate", help="Translate the entered text.")
    
    if translate_button and st.session_state.input_text.strip():
        with st.spinner("Translating..."):
            st.session_state.translated_text = translate_text(st.session_state.input_text, lang_direction)
            st.session_state.translated_text = add_spaces_between_sentences(st.session_state.translated_text)
            st.session_state.summarized_text = ""  # Clear previous summary

# Display translated text in a text area
if st.session_state.translated_text:
    st.subheader("Translated Text:")
    st.text_area("Translation:", value=st.session_state.translated_text, height=150, disabled=True)

# Summarize button
with col2:
    summarize_button = st.button("Summarize", key="summarize", help="Summarize the translated text.")
    
    if summarize_button and st.session_state.translated_text.strip():
        with st.spinner("Summarizing..."):
            st.session_state.summarized_text = summarize_text(st.session_state.translated_text)

# Display summarized text in a text area
if st.session_state.summarized_text:
    st.subheader("Summarized Text:")
    st.text_area("Summary:", value=st.session_state.summarized_text, height=150, disabled=True)

# Reset button for clearing the text areas
if st.button("Reset All"):
    st.session_state.input_text = ""
    st.session_state.translated_text = ""
    st.session_state.summarized_text = ""
    st.experimental_rerun()
