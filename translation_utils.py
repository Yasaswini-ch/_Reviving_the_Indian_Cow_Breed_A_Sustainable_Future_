# translation_utils.py (FINAL VERSION - Translation ONLY)

import streamlit as st
from googletrans import Translator
# Removed all imports related to audio (gTTS, os, io, hashlib)

# --- 1. Language Definitions ---
LANGUAGES = {
    'en': 'English',
    'hi': 'हिंदी (Hindi)',
    'bn': 'বাংলা (Bengali)',
    'te': 'తెలుగు (Telugu)',
    'mr': 'मराठी (Marathi)',
    'ta': 'தமிழ் (Tamil)',
    'gu': 'ગુજરાતી (Gujarati)',
    'kn': 'ಕನ್ನಡ (Kannada)',
    'ml': 'മലയാളം (Malayalam)',
    'pa': 'ਪੰਜਾਬੀ (Punjabi)',
    'ur': 'اردو (Urdu)',
    'or': 'ଓଡ଼ିଆ (Odia)', # Odia is included as googletrans supports its translation.
    'as': 'অসমীয়া (Assamese)',
    # Add more languages if needed
}

def safe_translate(text: str, dest_lang: str) -> str:
    """
    Translates text to the destination language using googletrans.
    Cached separately to avoid re-creating translator inside @st.cache_data.
    """
    if not text:
        return ""

    try:
        translator = Translator()
        translated = translator.translate(text, dest=dest_lang)
        return translated.text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text  # Return original on error


@st.cache_data(ttl=3600)
def translate_text(text: str, dest_lang: str) -> str:
    """
    Cached wrapper around the safe_translate function.
    Only caches pure inputs/outputs.
    """
    return safe_translate(text, dest_lang)


# --- 3. Placeholder for former Text-to-Speech (REMOVED) ---
# The text_to_speech function is completely removed from this file.

# --- 4. Streamlit Widget for Language Selection ---
def language_selector_widget(default_lang_code: str = 'en'):
    """
    Creates a centered dropdown for language selection.
    Returns the selected language code.
    """
    language_names = list(LANGUAGES.values())

    default_index = 0
    try:
        default_lang_name = LANGUAGES[default_lang_code]
        default_index = language_names.index(default_lang_name)
    except (KeyError, ValueError):
        default_index = 0

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        selected_lang_name = st.selectbox(
            "Choose Language:",
            options=language_names,
            index=default_index,
            key="selected_language_dropdown",
            help="Select the language for the content."
        )

    selected_lang_code = next(
        (code for code, name in LANGUAGES.items() if name == selected_lang_name),
        default_lang_code
    )
    return selected_lang_code

# --- 5. Wrapper for Streamlit text display functions (Translation ONLY) ---
def wrap_streamlit_text_function(st_func, current_lang_code):
    """
    Returns a wrapped version of a Streamlit text display function
    that automatically translates text. No audio functionality.
    """
    def wrapped_func(*args, **kwargs):
        if not args:
            return st_func(*args, **kwargs)

        original_text = args[0]
        if isinstance(original_text, str):
            translated_text = translate_text(original_text, current_lang_code)
            result = st_func(translated_text, *args[1:], **kwargs)
            # All audio-related calls (st.audio, text_to_audio) are removed.
            return result
        else:
            return st_func(*args, **kwargs) # Pass through if not a string
    return wrapped_func
