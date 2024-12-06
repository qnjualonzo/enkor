import streamlit as st
from googletrans import Translator
from pyAutoSummarizer.base import summarization
import re

# Initialize the translator object
translator = Translator()

# Initialize session state variables
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "summarized_text" not in st.session_state:
    st.session_state.summarized_text = ""
if "lang_direction" not in st.session_state:
    st.session_state.lang_direction = "EN to KO"
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Function to add spaces between sentences
def add_spaces_between_sentences(text):
    return re.sub(r'([.!?])(?=\S)', r'\1 ', text)

# Function to handle translation
def translate_text(input_text, src_lang, tgt_lang):
    try:
        translation = translator.translate(input_text, src=src_lang, dest=tgt_lang)
        return translation.text
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return ""

# Generalized summarization function
def summarize_text(text, lang, num_sentences=3):
    try:
        parameters = {
            'stop_words': [lang],
            'n_words': -1,
            'n_chars': -1,
            'lowercase': True,
            'rmv_accents': lang == 'en',  # Remove accents for English text
            'rmv_special_chars': lang == 'en',  # Remove special chars for English
            'rmv_numbers': False,
            'rmv_custom_words': [],
            'verbose': False
        }
        smr = summarization(text, **parameters)
        rank = smr.summ_ext_LSA(embeddings=False, model='all-MiniLM-L6-v2')
        return smr.show_summary(rank, n=num_sentences)
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return ""

# Streamlit UI Elements

# Set the title
st.title("EnKoreS: Translation and Summarization")

# Add custom styling for the app
st.markdown("""
<style>
    .title {
        font-size: 32px;
        font-weight: bold;
        color: #2E7D32;
    }
    .subheader {
        font-size: 18px;
        color: #0288D1;
    }
    .button {
        background-color: #388E3C;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

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
        src_lang = "en" if lang_direction == "EN to KO" else "ko"
        tgt_lang = "ko" if lang_direction == "EN to KO" else "en"
        with st.spinner("Translating..."):
            st.session_state.translated_text = translate_text(st.session_state.input_text, src_lang, tgt_lang)
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
        processed_text = add_spaces_between_sentences(st.session_state.translated_text)
        lang = 'ko' if lang_direction == "EN to KO" else 'en'
        with st.spinner("Summarizing..."):
            st.session_state.summarized_text = summarize_text(processed_text, lang)

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
