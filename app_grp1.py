import streamlit as st
from googletrans import Translator
from pyAutoSummarizer.base import summarization
import re

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

# Streamlit UI elements
st.title("EnKoreS")

lang_direction = st.sidebar.radio("Select Translation Direction", ["EN to KO", "KO to EN"])

# Reset input text and translations when the direction changes
if lang_direction != st.session_state.lang_direction:
    st.session_state.lang_direction = lang_direction
    st.session_state.input_text = ""
    st.session_state.translated_text = ""
    st.session_state.summarized_text = ""

# Input text area
st.session_state.input_text = st.text_area("Enter text to translate:", value=st.session_state.input_text)

# Translation button
if st.button("Translate"):
    if st.session_state.input_text.strip():
        src_lang = "en" if lang_direction == "EN to KO" else "ko"
        tgt_lang = "ko" if lang_direction == "EN to KO" else "en"
        st.session_state.translated_text = translate_text(st.session_state.input_text, src_lang, tgt_lang)
        st.session_state.translated_text = add_spaces_between_sentences(st.session_state.translated_text)
        st.session_state.summarized_text = ""  # Clear previous summary

# Display translated text
if st.session_state.translated_text:
    st.text_area("Translated Text:", value=st.session_state.translated_text, height=150, disabled=True)

    # Summarization button
    if st.button("Summarize"):
        if st.session_state.translated_text.strip():
            processed_text = add_spaces_between_sentences(st.session_state.translated_text)
            lang = 'ko' if lang_direction == "EN to KO" else 'en'
            st.session_state.summarized_text = summarize_text(processed_text, lang)

# Display summarized text
if st.session_state.summarized_text:
    st.text_area("Summarized Text:", value=st.session_state.summarized_text, height=150, disabled=True)
