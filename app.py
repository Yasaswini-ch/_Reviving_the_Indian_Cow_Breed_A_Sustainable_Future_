import streamlit as st
from streamlit_option_menu import option_menu
from googletrans import Translator, LANGUAGES
import requests
import pandas as pd
import google.generativeai as genai
from gtts import gTTS
import uuid
import os
import base64
import io
import random
from dotenv import load_dotenv
from PIL import Image # Ensure PIL is imported
import sqlite3
from datetime import datetime, timedelta, date # Specific imports
import bcrypt
import tensorflow as tf
from keras.preprocessing import image as keras_image
import cv2
import numpy as np
import supervision as sv
import uuid
import traceback
import logging
import json
from translation_utils import language_selector_widget, translate_text, wrap_streamlit_text_function, LANGUAGES

# --- NEW IMPORTS FOR HEALTH FEATURES & EXCEL ---
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO # for serving PDF/Excel in Streamlit
# import openpyxl # Pandas handles Excel writing with its own engine or openpyxl if installed

# --- Configuration ---
st.set_page_config(
    page_title="Kamadhenu Program",
    page_icon="🐄",
    layout="wide"
)

# --- Initialize Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if "current_lang" not in st.session_state:
    st.session_state["current_lang"] = 'en'

# --- 3. Centralized Language Selector ---
selected_language_code_from_widget = language_selector_widget(default_lang_code=st.session_state["current_lang"])

if st.session_state["current_lang"] != selected_language_code_from_widget:
    st.session_state["current_lang"] = selected_language_code_from_widget
    st.rerun()

current_lang = st.session_state["current_lang"]

class TranslatedStreamlit:
    def __init__(self, current_lang_code):
        # Streamlit text functions wrappers
        self.write = wrap_streamlit_text_function(st.write, current_lang_code)
        self.title = wrap_streamlit_text_function(st.title, current_lang_code)
        self.header = wrap_streamlit_text_function(st.header, current_lang_code)
        self.subheader = wrap_streamlit_text_function(st.subheader, current_lang_code)
        self.markdown = wrap_streamlit_text_function(st.markdown, current_lang_code)
        self.caption = wrap_streamlit_text_function(st.caption, current_lang_code)
        self.info = lambda text, icon: st.info(f"**{translate_text(text, current_lang_code)}**", icon=icon)
        self.success = lambda text, icon=None: st.success(f"**{translate_text(text, current_lang_code)}**", icon=icon) # Added success
        self.warning = lambda text, icon=None: st.warning(f"**{translate_text(text, current_lang_code)}**", icon=icon) # <--- ADD THIS LINE
        self.error = lambda text, icon=None: st.error(f"**{translate_text(text, current_lang_code)}**", icon=icon)     # <--- ADD THIS LINE (Good to have)

        # Internal dictionary for forum-specific string translations
        # These are used via ts['key'] syntax, as you've correctly updated in the forum code
        self._translations = {
            "community_forum_description": "Connect with fellow **Gaupalaks (Cattle Keepers)** and **Agricultural Experts**. Ask questions, share your valuable experiences, and help foster a thriving ecosystem of knowledge and prosperity for the Kamadhenu program.",
            "view_posts_in": "🌐 View posts in:",
            "create_new_post_header": "📢 Create a New Post / Ask an Expert",
            "your_name_username": "👤 Your Name / Username",
            "post_title_label": "📝 Post Title (e.g., Question about Gir Cow Feeding, Sharing Sahiwal Milking Tips)",
            "your_message_question": "💬 Your Message / Question",
            "select_category_optional": "Select Category (Optional)",
            "post_language_prompt": "In which language are you writing your post?",
            "tag_expert_question": "❓ Tag as a **Question for Experts** (Highlights your post for expert attention)",
            "post_message_button": "Post Message",
            "post_title_empty_warning": "Post Title and Message cannot be empty.",
            "post_submitted_success": "✅ Post submitted successfully to the Kamadhenu Community Network!",
            "recent_discussions_header": "Recent Discussions",
            "filter_by_category": "Filter by Category:",
            "search_posts": "Search Posts (Title or Text):",
            "show_expert_questions_only": "❓ Show Expert Questions Only",
            "no_discussions_found": "No discussions found matching your criteria. Be the first to start one in this category or clear filters!",
            "untitled_post": "Untitled Post",
            "category": "Category",
            "unknown_author": "Unknown",
            "upvote_button": "👍 Upvote ({count})",
            "replies_header": "###### Replies:",
            "reply_to_post": "💬 Reply to {author_name}",
            "your_reply": "💬 Your Reply",
            "reply_language_prompt": "In which language are you writing your reply?",
            "submit_reply_button": "Submit Reply",
            "reply_posted_success": "✅ Reply posted!",
            "reply_empty_warning": "Reply message cannot be empty.",
            "respectful_discussions_note": "<br><sub>Keep discussions respectful, constructive, and relevant to the Kamadhenu Program's focus.</sub>",
            "general_discussion": "General Discussion",
            "breed_specific": "Breed Specific",
            "feeding_nutrition": "Feeding & Nutrition",
            "health_disease": "Health & Disease",
            "farming_practices": "Farming Practices",
            "government_schemes": "Government Schemes",
            "market_sales": "Market & Sales",
            "machinery_equipment": "Machinery & Equipment",
            "other": "Other",
            "all_categories": "All Categories",
            "expert_question_label": "💡 **Question for Experts**",
            "author_label": "👤 {author_name}",
            "timestamp_label": "🕒 *{timestamp}*",
            "language_label": "🌐 *({language})*"
        }

    def __getitem__(self, key):
        """Allows accessing general translations like ts['community_forum_title']"""
        if key in self._translations:
            # For these strings, we want to *always* translate them to current_lang
            # as they are hardcoded text within the app logic.
            return translate_text(self._translations[key], st.session_state["current_lang"])
        logger.warning(f"Missing translation key: {key}")
        return f"MISSING_TRANSLATION_KEY_{key}"

# This should be the ONLY instantiation of ts in your entire script.
ts = TranslatedStreamlit(current_lang)
# Now, 'current_lang' from session state is the definitive selected language
current_lang = st.session_state["current_lang"]
# --- Load Environment Variables & API Keys ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
# BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/predict/") # Not used if Roboflow is direct

# --- Roboflow Configuration ---
ROBOFLOW_PROJECT_ID = "cattle-breed-9rfl6-xqimv-mqao3" # Your specific project
ROBOFLOW_MODEL_VERSION = 6 # Your specific version
CONFIDENCE_THRESHOLD = 40
OVERLAP_THRESHOLD = 30

# --- Initialize Google Generative AI API ---
gemini_model = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        try:
            gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
            logger.info("Google AI Model (gemini-1.5-flash-latest) initialized.")
        except Exception as model_err:
             st.warning(f"Could not initialize Google AI Model: {model_err}. Chatbot might not function.", icon="⚠️")
             logger.warning(f"Google AI Model initialization failed: {model_err}")
             gemini_model = None
    except Exception as e:
        st.error(f"Error configuring Google AI SDK: {e}")
        logger.error(f"Google AI SDK Config Error: {e}\n{traceback.format_exc()}")
        GOOGLE_API_KEY = None
else:
    if os.path.exists(".env"):
         st.warning("Google API key not found in .env! Chatbot requires GOOGLE_API_KEY.", icon="🔑")
    else:
         st.warning(".env file not found. Chatbot requires GOOGLE_API_KEY.", icon="📄")

# --- Initialize Roboflow Model (Cached) ---
@st.cache_resource
def load_roboflow_model():
    if not ROBOFLOW_API_KEY:
        # st.error("Roboflow API Key (ROBOFLOW_API_KEY) not found. Breed identification disabled.", icon="🔑")
        logger.warning("Roboflow API Key not found. Breed ID disabled.")
        return None
    try:
        from roboflow import Roboflow # Import here to avoid issues if not installed globally
        logger.info("Initializing Roboflow (cached)...")
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace().project(ROBOFLOW_PROJECT_ID)
        model = project.version(ROBOFLOW_MODEL_VERSION).model
        logger.info("Roboflow model loaded successfully (cached).")
        return model
    except ImportError:
        logger.error("Roboflow library not installed. Breed identification disabled.")
        # st.error("Roboflow library not found. Please install it: pip install roboflow")
        return None
    except Exception as e:
        # st.error(f"Failed to initialize Roboflow model: {e}. Breed identification disabled.")
        logger.error(f"Roboflow Initialization Error: {e}\n{traceback.format_exc()}")
        return None
roboflow_model = load_roboflow_model()

SKIN_DISEASE_MODEL_PATH = r"C:\Users\chebo\Kamdhenu_App-main\model\model.h5"
# Relative to app1.py
CLASS_NAMES_SKIN_DISEASE = ["Bacterial", "Fungal", "Healthy"] # Must match model output order

DISEASE_INFO_AND_ADVICE_SKIN = {
    "Bacterial": {
        "common_examples": "Examples include Dermatophilosis (Rain Scald), Staphylococcal infections (Pyoderma).",
        "general_appearance": "Often present as pustules (pus-filled bumps), moist/oozing lesions, crusts, or abscesses. The skin might be inflamed and painful.",
        "possible_factors": "Skin injuries, prolonged wetness, poor hygiene, insect bites, or a compromised immune system can be predisposing factors.",
        "advice": [
            "**Consult a veterinarian immediately for accurate diagnosis and appropriate antibiotic treatment.**",
            "Isolate the affected animal to prevent potential spread to others.",
            "Keep the affected area clean and dry as much as possible (follow vet guidance).",
            "Improve overall hygiene in the animal's living environment.",
            "Do not apply any ointments or medications without veterinary prescription."
        ]
    },
    "Fungal": {
        "common_examples": "The most common is Dermatophytosis (Ringworm).",
        "general_appearance": "Typically appears as circular, scaly, hairless patches. Lesions may be slightly raised, crusted, and sometimes itchy. Can occur anywhere on the body.",
        "possible_factors": "Often spread by direct contact with infected animals or contaminated environments/equipment. Young, old, or immunocompromised animals are more susceptible. Humid conditions can favor growth.",
        "advice": [
            "**Consult a veterinarian for confirmation and appropriate antifungal treatment (topical and/or systemic).**",
            "Ringworm is zoonotic (can spread to humans) and highly contagious to other animals. Isolate the affected animal strictly.",
            "Thoroughly clean and disinfect the animal's environment, grooming tools, and any items it contacted.",
            "Wear gloves when handling the affected animal or cleaning its environment.",
            "Ensure good ventilation and dry conditions for all animals."
        ]
    },
    "Healthy": {
        "common_examples": "No signs of common bacterial or fungal infections detected in the image.",
        "general_appearance": "Skin appears clear, without unusual lesions, scaling, or significant hair loss in the area shown.",
        "possible_factors": "Good hygiene, balanced nutrition, and a stress-free environment contribute to healthy skin.",
        "advice": [
            "The AI suggests the skin in the image appears healthy based on its training.",
            "Continue good management practices and regular observation of all animals for any changes in skin health.",
            "Remember, this tool only analyzes the provided image. For a complete health assessment, always rely on regular veterinary check-ups."
        ]
    }
}

@st.cache_resource
def load_skin_disease_model():
    if not os.path.exists(SKIN_DISEASE_MODEL_PATH):
        logger.error(f"Skin disease prediction model not found at: {SKIN_DISEASE_MODEL_PATH}")
        st.error(f"🐛 Skin disease prediction model not found. Please ensure '{SKIN_DISEASE_MODEL_PATH}' exists.", icon="MODEL")
        return None
    try:
        model = tf.keras.models.load_model(SKIN_DISEASE_MODEL_PATH)
        logger.info("Skin disease prediction model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading skin disease prediction model: {e}", exc_info=True)
        st.error(f"🐛 Error loading skin disease prediction model: {e}")
        return None

def preprocess_skin_image(img_pil: Image.Image):
    img_resized = img_pil.resize((150, 150))
    img_array = keras_image.img_to_array(img_resized)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_normalized = img_array_expanded / 255.0
    return img_normalized

def render_skin_disease_detector_ui():
    ts.subheader("🔬 Image-Based Skin Disease Detector (Beta)")
    ts.markdown("Upload a clear close-up image of the affected skin area for a preliminary AI-based prediction.")
    st.warning("**Disclaimer:** This tool provides an initial suggestion and is **NOT** a substitute for professional veterinary diagnosis. Always consult a qualified veterinarian for any health concerns.", icon="⚠️")

    uploaded_skin_image = st.file_uploader("Upload Skin Image (JPG, PNG, JPEG)...", type=['png','jpg','jpeg'], key="skin_disease_uploader")

    if uploaded_skin_image is None:
        st.info("Please upload an image of the affected skin area to get a prediction.")
        return

    try:
        img_pil = Image.open(uploaded_skin_image).convert("RGB")
        st.image(img_pil, caption="Uploaded Skin Image", width=300)

        processed_tensor = preprocess_skin_image(img_pil)
        model = load_skin_disease_model()

        if model is None: return

        with st.spinner("🧠 Analyzing skin image..."):
            predictions = model.predict(processed_tensor)

        predicted_class_index = np.argmax(predictions[0])
        confidence_score = predictions[0][predicted_class_index]
        predicted_class_name = CLASS_NAMES_SKIN_DISEASE[predicted_class_index]

        st.success(f"**Predicted Condition: {predicted_class_name}** ({confidence_score:.1%} confidence)")

        if predicted_class_name in DISEASE_INFO_AND_ADVICE_SKIN:
            info = DISEASE_INFO_AND_ADVICE_SKIN[predicted_class_name]
            with st.container(border=True):
                ts.markdown(f"##### General Information about Potential '{predicted_class_name}' Conditions:")
                if "common_examples" in info: st.markdown(f"**Common Examples:** {info['common_examples']}")
                if "general_appearance" in info: st.markdown(f"**General Appearance:** {info['general_appearance']}")
                if "possible_factors" in info: st.markdown(f"**Possible Contributing Factors:** {info['possible_factors']}")
                ts.markdown("##### Recommended Actions & Advice:")
                for point in info["advice"]:
                    if "**Consult a veterinarian immediately**" in point: st.error(point, icon="👩‍⚕️")
                    elif predicted_class_name == "Healthy" and "AI suggests" in point: st.info(point, icon="💡")
                    else: st.markdown(f"- {point}")
        if predicted_class_name != "Healthy":
            st.error("**Important:** This AI prediction is for informational purposes only. Crucially, consult a qualified veterinarian for an accurate diagnosis and treatment plan.", icon="⚠️")
        else:
            st.info("While the AI suggests the area in the image appears healthy, continue regular observation and consult your vet for routine checks or if you notice any changes.", icon="ℹ️")

    except Exception as e:
        logger.error(f"Error during skin disease prediction: {e}", exc_info=True)
        st.error(f"An error occurred: {e}. Ensure image is valid (PNG, JPG, JPEG).")
        

# --- GENERAL Community Forum Functions ---
COMMUNITY_FORUM_POST_FILE = "community_forum_posts.json" # General forum posts

def load_main_forum_posts():
    if os.path.exists(COMMUNITY_FORUM_POST_FILE):
        try:
            with open(COMMUNITY_FORUM_POST_FILE, "r", encoding="utf-8") as f:
                posts = json.load(f)
                for i, post in enumerate(posts):
                    if "id" not in post: post["id"] = post.get("timestamp", str(uuid.uuid4())) + "_" + str(i)
                    if "replies" not in post: post["replies"] = []
                    if "author" not in post and "name" in post : post["author"] = post["name"]
                    # Changed to ts["key"]
                    if "title" not in post : post["title"] = ts["general_discussion"] 
                    if "tags" not in post: post["tags"] = [] # For expert tagging
                    if "language" not in post: post["language"] = "English" # For language selection
                    for reply in post.get("replies", []):
                        if "author" not in reply and "name" in reply: reply["author"] = reply["name"]

                return posts
        except json.JSONDecodeError:
            logger.warning(f"Forum file '{COMMUNITY_FORUM_POST_FILE}' corrupted. Starting fresh.")
            return []
        except Exception as e:
            logger.error(f"Error loading main forum posts: {e}")
            return []
    return []

def save_main_forum_posts(posts_list):
    try:
        with open(COMMUNITY_FORUM_POST_FILE, "w", encoding="utf-8") as f:
            json.dump(posts_list, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.toast(f"Error saving main forum posts: {e}", icon="🔥")
        logger.error(f"Error saving main forum posts: {e}")


def render_main_community_forum_ui():
    st.title("🤝 Kamadhenu Community Network") # Changed from ts.key to ts["key"]
    st.markdown(ts["community_forum_description"]) # Changed from ts.key to ts["key"]

    if "main_forum_posts" not in st.session_state:
        st.session_state.main_forum_posts = load_main_forum_posts()

    if "main_forum_user_message" in st.session_state and st.session_state.main_forum_user_message:
        msg_data = st.session_state.main_forum_user_message
        msg_type = msg_data.get("type", "info")
        if msg_data.get("text"):
            if msg_type == "success": st.success(msg_data["text"])
            elif msg_type == "warning": st.warning(msg_data["text"])
            elif msg_type == "error": st.error(msg_data["text"])
            else: st.info(msg_data["text"])
        del st.session_state.main_forum_user_message

    st.markdown("---")
    view_language = st.selectbox(
        ts["view_posts_in"], # Changed to ts["key"]
        options=["English", "Hindi", "Telugu"], # Add more local languages as desired
        key="main_forum_view_language"
    )
    st.markdown("---")

    with st.expander(ts["create_new_post_header"], expanded=not bool(st.session_state.main_forum_posts)): # Changed to ts["key"]
        with st.form("main_forum_post_form", clear_on_submit=True):
            author_name_forum = st.text_input(ts["your_name_username"], key="main_forum_author", # Changed to ts["key"]
                                            value=st.session_state.get("username", "Guest") if st.session_state.get("logged_in") else "Guest")
            post_title_forum = st.text_input(ts["post_title_label"], key="main_forum_title") # Changed to ts["key"]
            post_text_forum = st.text_area(ts["your_message_question"], height=150, key="main_forum_text") # Changed to ts["key"]
            
            col1, col2 = st.columns(2)
            post_category_forum = col1.selectbox(ts["select_category_optional"], # Changed to ts["key"]
                                            options=[
                                                ts["general_discussion"], ts["breed_specific"], ts["feeding_nutrition"], # Changed to ts["key"]
                                                ts["health_disease"], ts["farming_practices"], ts["government_schemes"], # Changed to ts["key"]
                                                ts["market_sales"], ts["machinery_equipment"], ts["other"] # Changed to ts["key"]
                                            ],
                                            key="main_forum_category")
            
            post_language_forum = col2.selectbox(
                ts["post_language_prompt"], # Changed to ts["key"]
                options=["English", "Hindi", "Telugu"], # Allow users to specify their post language
                key="main_forum_post_language"
            )

            is_question_for_expert = st.checkbox(ts["tag_expert_question"], key="main_forum_expert_tag") # Changed to ts["key"]
            
            submitted_forum = st.form_submit_button(ts["post_message_button"]) # Changed to ts["key"]

            if submitted_forum:
                final_author_forum = author_name_forum.strip() or "Anonymous"
                if post_text_forum.strip() and post_title_forum.strip():
                    new_post_forum = {
                        "id": str(uuid.uuid4()),
                        "author": final_author_forum,
                        "title": post_title_forum.strip(),
                        "text": post_text_forum.strip(),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "category": post_category_forum,
                        "language": post_language_forum,
                        "tags": ["expert_question"] if is_question_for_expert else [],
                        "replies": [],
                        "upvotes": 0
                    }
                    st.session_state.main_forum_posts.insert(0, new_post_forum)
                    save_main_forum_posts(st.session_state.main_forum_posts)
                    st.session_state.main_forum_user_message = {"type": "success", "text": ts["post_submitted_success"]} # Changed to ts["key"]
                    st.rerun()
                else:
                    st.warning(ts["post_title_empty_warning"]) # Changed to ts["key"]

    st.markdown("---")
    st.subheader(ts["recent_discussions_header"]) # Changed to ts["key"]

    # Filtering options
    forum_filter_col1, forum_filter_col2, forum_filter_col3 = st.columns([1, 1, 0.8])
    all_categories = [ts["all_categories"]] + sorted(list(set(p.get("category", ts["general_discussion"]) for p in st.session_state.main_forum_posts))) # Changed to ts["key"]
    selected_category_filter = forum_filter_col1.selectbox(ts["filter_by_category"], all_categories, key="main_forum_category_filter") # Changed to ts["key"]
    search_term_forum = forum_filter_col2.text_input(ts["search_posts"], key="main_forum_search") # Changed to ts["key"]
    
    show_expert_questions = forum_filter_col3.checkbox(ts["show_expert_questions_only"], key="main_forum_expert_filter") # Changed to ts["key"]


    filtered_posts_forum = st.session_state.main_forum_posts
    if selected_category_filter != ts["all_categories"]: # Changed to ts["key"]
        filtered_posts_forum = [p for p in filtered_posts_forum if p.get("category") == selected_category_filter]
    if search_term_forum:
        search_lower = search_term_forum.lower()
        filtered_posts_forum = [p for p in filtered_posts_forum if search_lower in p.get("title","").lower() or search_lower in p.get("text","").lower()]
    if show_expert_questions:
        filtered_posts_forum = [p for p in filtered_posts_forum if "expert_question" in p.get("tags", [])]


    if not filtered_posts_forum:
        st.info(ts["no_discussions_found"]) # Changed to ts["key"]
    else:
        # Display posts
        for post_forum in list(filtered_posts_forum):
            with st.container(border=True):
                # Using .format() for strings with placeholders, accessing via ts["key"]
                st.markdown(f"##### {post_forum.get('title', ts['untitled_post'])} <small> ({ts['category']}: {post_forum.get('category', 'N/A')})</small>", unsafe_allow_html=True)
                
                if "expert_question" in post_forum.get("tags", []):
                    st.markdown(ts["expert_question_label"], unsafe_allow_html=True) # Changed to ts["key"]
                
                st.markdown(ts["author_label"].format(author_name=post_forum.get('author', ts["unknown_author"])) + # Changed to ts["key"].format()
                            f"    <small>" + ts["timestamp_label"].format(timestamp=post_forum.get('timestamp','N/A')) + # Changed to ts["key"].format()
                            ts["language_label"].format(language=post_forum.get('language', 'English')) + "</small>", unsafe_allow_html=True) # Changed to ts["key"].format()
                
                st.write(post_forum.get("text","")) 
                
                if st.button(ts["upvote_button"].format(count=post_forum.get('upvotes', 0, key="auto_btn_0")), key=f"upvote_{post_forum['id']}"): # Changed to ts["key"].format()
                    post_forum['upvotes'] = post_forum.get('upvotes', 0) + 1
                    save_main_forum_posts(st.session_state.main_forum_posts)
                    st.rerun()

                if post_forum.get("replies"):
                    st.markdown(ts["replies_header"]) # Changed to ts["key"]
                    for reply_forum in post_forum["replies"]:
                        st.markdown(f"   ↪️ **{reply_forum.get('author',ts['unknown_author'])}** ({ts['timestamp_label'].format(timestamp=reply_forum.get('timestamp','N/A'))} {ts['language_label'].format(language=reply_forum.get('language', 'English'))}): {reply_forum.get('text','')}") # Changed to ts["key"].format()

                with st.expander(ts["reply_to_post"].format(author_name=post_forum.get('author','this post')), expanded=False): # Changed to ts["key"].format()
                    reply_form_key_forum = f"main_forum_reply_form_{post_forum['id']}"
                    with st.form(key=reply_form_key_forum, clear_on_submit=True):
                        reply_author_forum = st.text_input(ts["your_name_username"], key=f"main_forum_reply_author_{post_forum['id']}", # Changed to ts["key"]
                                                            value=st.session_state.get("username", "Guest") if st.session_state.get("logged_in") else "Guest")
                        reply_text_forum = st.text_area(ts["your_reply"], height=75, key=f"main_forum_reply_text_{post_forum['id']}") # Changed to ts["key"]
                        
                        reply_language_forum = st.selectbox(
                            ts["reply_language_prompt"], # Changed to ts["key"]
                            options=["English", "Hindi", "Telugu"],
                            key=f"main_forum_reply_language_{post_forum['id']}"
                        )

                        submit_reply_forum = st.form_submit_button(ts["submit_reply_button"]) # Changed to ts["key"]

                        if submit_reply_forum:
                            final_reply_author_forum = reply_author_forum.strip() or "Anonymous"
                            if reply_text_forum.strip():
                                new_reply_forum = {
                                    "author": final_reply_author_forum,
                                    "text": reply_text_forum.strip(),
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "language": reply_language_forum
                                }
                                for p_loop_forum in st.session_state.main_forum_posts:
                                    if p_loop_forum['id'] == post_forum['id']:
                                        p_loop_forum['replies'].append(new_reply_forum)
                                        break
                                save_main_forum_posts(st.session_state.main_forum_posts)
                                st.session_state.main_forum_user_message = {"type": "success", "text": ts["reply_posted_success"]} # Changed to ts["key"]
                                st.rerun()
                            else:
                                st.warning(ts["reply_empty_warning"]) # Changed to ts["key"]
            st.markdown("---")
    st.markdown(ts["respectful_discussions_note"], unsafe_allow_html=True) # Changed to ts["key"]


# --- Database Connection ---
DB_FILE = 'Cows.db'
@st.cache_resource
def get_connection():
    try:
        logger.info(f"Connecting to database: {DB_FILE}")
        return sqlite3.connect(DB_FILE, check_same_thread=False, timeout=10)
    except sqlite3.Error as e:
        st.error(f"Database connection error: {e}")
        logger.error(f"Database Connection Error: {e}\n{traceback.format_exc()}")
        return None

# --- Session State Initialization ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
# ... (other session state variables) ...
if 'current_page' not in st.session_state: st.session_state.current_page = "Home"

# NEW session state variables for detailed view
if 'viewing_listing_id' not in st.session_state: st.session_state.viewing_listing_id = None
if 'viewing_listing_type' not in st.session_state: st.session_state.viewing_listing_type = None # 'cattle' or 'machinery'
if 'previous_page' not in st.session_state: st.session_state.previous_page = "Home" # To navigate back

# --- Helper Functions ---

# --- Helper function to save uploaded images ---
def save_uploaded_image(uploaded_file, subfolder):
    """Saves an uploaded image to a specified subfolder and returns its relative path."""
    if uploaded_file is not None:
        # Create the main upload directory if it doesn't exist
        base_upload_dir = "uploaded_images"
        if not os.path.exists(base_upload_dir):
            os.makedirs(base_upload_dir)

        # Create the subfolder if it doesn't exist
        target_folder = os.path.join(base_upload_dir, subfolder)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # Generate a unique filename to prevent overwrites
        file_extension = os.path.splitext(uploaded_file.name)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(target_folder, unique_filename)

        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logger.info(f"Image saved to: {file_path}")
            return os.path.join(subfolder, unique_filename) # Return relative path from 'uploaded_images'
        except Exception as e:
            st.error(f"Error saving image {uploaded_file.name}: {e}")
            logger.error(f"Error saving image {uploaded_file.name}: {e}")
            return None
    return None

# --- Helper Functions (Continued) ---

# ... (your existing load_image, display_image, save_uploaded_image, etc.) ...

@st.cache_resource
def load_skin_disease_model():
    if not os.path.exists(SKIN_DISEASE_MODEL_PATH):
        logger.error(f"Skin disease prediction model not found at: {SKIN_DISEASE_MODEL_PATH}")
        # Don't use st.error here as it will show on every page load if model is missing
        # The UI function will handle the error message.
        return None
    try:
        model = tf.keras.models.load_model(SKIN_DISEASE_MODEL_PATH)
        logger.info("Skin disease prediction model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading skin disease prediction model: {e}", exc_info=True)
        # UI function will handle error message
        return None

def preprocess_skin_image(img_pil: Image.Image): # Ensure PIL.Image is imported
    # Ensure img_pil is a PIL Image object
    if not isinstance(img_pil, Image.Image):
        raise ValueError("Input to preprocess_skin_image must be a PIL.Image object.")
    img_resized = img_pil.resize((150, 150)) # Common size for many image models
    img_array = keras_image.img_to_array(img_resized)
    img_array_expanded = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_normalized = img_array_expanded / 255.0 # Normalize to 0-1 range
    return img_normalized

# Attempt to load the model once at the start to see if it's available
# The UI will still call load_skin_disease_model() which uses the cache.
skin_model_global = load_skin_disease_model()

# --- Helper function to display uploaded images (adjusting for relative paths) ---
def display_uploaded_image(relative_image_path, caption="", width=150):
    """Displays an image from the 'uploaded_images' folder."""
    if relative_image_path:
        full_image_path = os.path.join("uploaded_images", relative_image_path)
        if os.path.exists(full_image_path):
            try:
                img = Image.open(full_image_path)
                st.image(img, caption=caption, width=width)
            except Exception as e:
                logger.error(f"Error displaying uploaded image {full_image_path}: {e}")
                st.caption(f"Error loading image: {os.path.basename(relative_image_path)}")
        else:
            logger.warning(f"display_uploaded_image: File not found at {full_image_path}")
            # st.caption(f"Image not found: {os.path.basename(relative_image_path)}") # Optional: show in UI

@st.cache_data
def load_image(image_path):
    """Loads an image using PIL, returns None if path is invalid."""
    full_path = os.path.join("images", os.path.basename(image_path))
    if os.path.exists(full_path):
        try:
            return Image.open(full_path)
        except Exception as e:
            logger.error(f"Error loading image {full_path}: {e}")
            return None
    else:
        # Log if image not found, but don't show st.warning in UI unless necessary
        logger.warning(f"Helper: Image file not found at path: {full_path}")
        return None
    
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

@st.cache_data # Cache geocoding results
def get_lat_lon_from_address(address_str):
    geolocator = Nominatim(user_agent="kamdhenu_app_v1") # Be respectful with user_agent
    try:
        location = geolocator.geocode(address_str, timeout=10)
        if location:
            return location.latitude, location.longitude
    except GeocoderTimedOut:
        logger.warning(f"Geocoding timed out for address: {address_str}")
        return None, None
    except GeocoderUnavailable:
        logger.warning(f"Geocoding service unavailable for address: {address_str}")
        return None, None
    except Exception as e:
        logger.error(f"Geocoding error for {address_str}: {e}")
        return None, None
    return None, None

# Example usage when a farmer saves their profile:
# if st.form_submit_button("Save Profile"):
#     full_address = f"{farmer_street_address}, {farmer_city}, {farmer_state}, {farmer_pincode}, India" # Construct full address
#     lat, lon = get_lat_lon_from_address(full_address)
#     if lat and lon:
#         # Update user's record in DB with lat, lon
#         cursor.execute("UPDATE users SET latitude=?, longitude=? WHERE user_id=?", (lat, lon, st.session_state.user_id))
#         conn.commit()
#         st.success("Profile updated with location!")
#     else:
#         st.warning("Could not determine location from address. Please check address or try again later.")

def get_donation_campaigns_db(conn):
    """
    Fetches all active donation campaigns from the 'donation_campaigns' table.
    Expects a database connection object.
    The fetched rows will now contain 'campaign_id' instead of 'id'.
    """
    cursor = conn.cursor()
    try:
        # No change in SELECT * FROM, but the returned row structure now has 'campaign_id'
        cursor.execute("SELECT * FROM donation_campaigns WHERE is_active = 1")
        return cursor.fetchall()
    except sqlite3.Error as e:
        st.error(f"Error fetching donation campaigns: {e}")
        return []

def update_campaign_donation_db(conn, campaign_id, amount):
    """
    Updates the 'current_amount' for a specific donation campaign.
    Uses 'campaign_id' for the WHERE clause.
    """
    cursor = conn.cursor()
    try:
        # Changed 'id = ?' to 'campaign_id = ?'
        cursor.execute("UPDATE donation_campaigns SET current_amount = current_amount + ? WHERE campaign_id = ?",
                       (amount, campaign_id))
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Error updating campaign donation: {e}")
        return False

def get_adoptable_cows_db(conn):
    """
    Fetches all cows with a 'status' of 'Available' from the 'adoptable_cows' table.
    The fetched rows will now contain 'cow_id' instead of 'id'.
    """
    cursor = conn.cursor()
    try:
        # No change in SELECT * FROM, but the returned row structure now has 'cow_id'
        cursor.execute("SELECT * FROM adoptable_cows WHERE status = 'Available'")
        return cursor.fetchall()
    except sqlite3.Error as e:
        st.error(f"Error fetching adoptable cows: {e}")
        return []

def adopt_cow_db(conn, cow_id):
    """
    Changes the 'status' of a cow to 'Adopted' in the 'adoptable_cows' table.
    Uses 'cow_id' for the WHERE clause.
    """
    cursor = conn.cursor()
    try:
        # Changed 'id = ?' to 'cow_id = ?'
        cursor.execute("UPDATE adoptable_cows SET status = 'Adopted' WHERE cow_id = ?", (cow_id,))
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Error adopting cow: {e}")
        return False

def log_mock_donation_db(conn, donor_name, amount, campaign_id=None, cow_id=None):
    """
    Logs a mock donation transaction in the 'mock_donations' table.
    The 'campaign_id' and 'cow_id' parameters directly map to the foreign key columns.
    The 'donation_log_id' primary key is auto-incremented.
    """
    cursor = conn.cursor()
    try:
        # The column names 'campaign_id' and 'cow_id' in the INSERT statement
        # directly match the new foreign key column names in the mock_donations table.
        cursor.execute("INSERT INTO mock_donations (donor_name, amount, campaign_id, cow_id, timestamp) VALUES (?, ?, ?, ?, ?)",
                       (donor_name, amount, campaign_id, cow_id, datetime.datetime.now()))
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Error logging mock donation: {e}")
        return False

# --- Helper Functions ---
UPLOAD_FOLDER = "uploaded_images" # Define this globally once

@st.cache_data # For static images
def load_static_image(image_filename): # Renamed for clarity
    """Loads a static image from the 'images' folder."""
    full_path = os.path.join("images", os.path.basename(image_filename)) # Assuming 'images' is the folder for static app images
    if os.path.exists(full_path):
        try: return Image.open(full_path)
        except Exception as e: logger.error(f"Error loading static image {full_path}: {e}"); return None
    else: logger.warning(f"Static image file not found: {full_path}"); return None

def display_static_image(image_filename, caption="", width=None, use_container_width=True):
    """Displays a static image from the 'images' folder."""
    img = load_static_image(image_filename)
    if img:
        st.image(img, caption=caption, use_container_width=use_container_width if width is None else False, width=width)
    elif image_filename:
        st.warning(f"Static image not found: {os.path.basename(image_filename)}", icon="🖼️")

# Helper for user-uploaded images
def save_uploaded_image(uploaded_file, subfolder):
    # ... (your existing save_uploaded_image function - seems okay) ...
    if uploaded_file is not None:
        target_subfolder = os.path.join(UPLOAD_FOLDER, subfolder)
        if not os.path.exists(target_subfolder): os.makedirs(target_subfolder)
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in ['.png', '.jpg', '.jpeg']:
            st.error(f"Invalid file type: {file_extension}. Upload JPG or PNG."); return None
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(target_subfolder, unique_filename)
        try:
            with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
            logger.info(f"Image saved: {file_path}"); return os.path.join(subfolder, unique_filename)
        except Exception as e: st.error(f"Error saving image: {e}"); logger.error(f"Err saving img: {e}"); return None
    return None


def display_uploaded_image(relative_image_path, caption="", width=None, use_container_width=False):
    """Displays an image from the UPLOAD_FOLDER."""
    if relative_image_path:
        full_image_path = os.path.join(UPLOAD_FOLDER, relative_image_path) # Use global UPLOAD_FOLDER
        if os.path.exists(full_image_path):
            try:
                img = Image.open(full_image_path)
                st.image(img, caption=caption,
                         width=width if not use_container_width else None,
                         use_container_width=use_container_width)
            except Exception as e:
                logger.error(f"Error displaying uploaded image {full_image_path}: {e}")
                st.caption(f"Err displaying: {os.path.basename(relative_image_path)}")
        else:
            logger.warning(f"display_uploaded_image: File not found at {full_image_path}")
    # else:
        # st.caption("No image")

def display_image(image_path, caption="", width=None, use_container_width=True):
    """Safely displays an image if it exists using st.image."""
    img = load_image(image_path)
    if img:
        st.image(img, caption=caption, use_container_width=use_container_width if width is None else False, width=width)

    elif image_path:
        # Use logger instead of st.warning to avoid cluttering UI for optional images
        logger.warning(f"display_image: Image not found: {os.path.basename(image_path)}")
        st.warning(f"Image not found: {os.path.basename(image_path)}", icon="🖼️")

# --- Helper function to check if calf_rearing_log table exists (place with other helpers) ---
def calf_rearing_table_exists(cursor):
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='calf_rearing_log';")
        return cursor.fetchone() is not None
    except sqlite3.Error as e:
        logger.error(f"Error checking calf_rearing_log table: {e}")
        return False

# --- Cattle Breed Data (Ensure image filenames match those in your ./images folder) ---
CATTLE_BREEDS_DATA = [
    {"name": "Gir", "region": "Gujarat", "milk_yield": 12, "strength": "High", "lifespan": 18, "image": "gir.jpg"},
    {"name": "Sahiwal", "region": "Punjab", "milk_yield": 14, "strength": "Medium", "lifespan": 20, "image": "sahiwal.jpg"},
    {"name": "Ongole", "region": "Andhra Pradesh", "milk_yield": 10, "strength": "Very High", "lifespan": 22, "image": "ongole.jpg"},
    {"name": "Punganur", "region": "Andhra Pradesh", "milk_yield": 6, "strength": "Low", "lifespan": 15, "image": "punganur.jpg"},
    {"name": "Amrit Mahal", "region": "Karnataka", "milk_yield": 7, "strength": "High", "lifespan": 18, "image": "amritmahal.jpg"}, # Check spelling: amritmahal.jpg
    {"name": "Deoni", "region": "Maharashtra", "milk_yield": 9, "strength": "Medium", "lifespan": 19, "image": "deoni.jpeg"},
    {"name": "Hallikar", "region": "Karnataka", "milk_yield": 8, "strength": "Very High", "lifespan": 20, "image": "hallikar.jpg"},
    {"name": "Kankrej", "region": "Gujarat", "milk_yield": 11, "strength": "High", "lifespan": 21, "image": "kankrej.jpg"},
    {"name": "Krishna Valley", "region": "Karnataka", "milk_yield": 7, "strength": "Very High", "lifespan": 19, "image": "krishna_valley.jpg"},
    {"name": "Malnad Gidda", "region": "Karnataka", "milk_yield": 5, "strength": "Medium", "lifespan": 16, "image": "malnad_gidda.jpeg"},
    {"name": "Rathi", "region": "Rajasthan", "milk_yield": 10, "strength": "Medium", "lifespan": 20, "image": "rathi.jpg"},
    {"name": "Red Sindhi", "region": "Sindh (Origin)", "milk_yield": 11, "strength": "High", "lifespan": 22, "image": "red_sindhi.jpg"},
    {"name": "Tharparkar", "region": "Rajasthan", "milk_yield": 9, "strength": "Medium", "lifespan": 21, "image": "tharparkar.jpg"}
]

# --- Weather API Configuration & Helper Functions ---
OPEN_METEO_API_URL = "https://api.open-meteo.com/v1/forecast"
@st.cache_data(ttl=3600)
def get_coordinates(city_name="Nagpur"):
    cities_coords = { "Delhi": {"latitude": 28.6139, "longitude": 77.2090}, "Mumbai": {"latitude": 19.0760, "longitude": 72.8777}, "Bengaluru": {"latitude": 12.9716, "longitude": 77.5946}, "Nagpur": {"latitude": 21.1458, "longitude": 79.0882} } # Add more
    return cities_coords.get(city_name, cities_coords["Nagpur"])
@st.cache_data(ttl=1800)
def fetch_weather_forecast(latitude, longitude):
    params = { "latitude": latitude, "longitude": longitude, "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,weather_code", "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max", "timezone": "auto", "forecast_days": 7 }
    try:
        response = requests.get(OPEN_METEO_API_URL, params=params, timeout=10); response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: st.toast(f"Weather API Error: {e}", icon="🌦️"); logger.error(f"Weather API Error: {e}"); return None
def interpret_weather_code(code):
    if code == 0: return "Clear"
    if code in [1, 2, 3]: return "Partly Cloudy"
    if code in [45, 48]: return "Fog"
    if code in [51, 53, 55, 56, 57]: return "Drizzle"
    if code in [61, 63, 65, 66, 67]: return "Rain"
    if code in [71, 73, 75, 77]: return "Snow"
    if code in [80, 81, 82]: return "Rain Showers"
    if code in [85, 86]: return "Snow Showers"
    if code in [95, 96, 99]: return "Thunderstorm"
    return f"Code {code}"
def generate_cattle_care_advice(daily_forecast_today, daily_forecast_next_days):
    advice, notifications = [], []
    if daily_forecast_today:
        today_max, today_min, precip_sum, precip_prob, weather_desc = (daily_forecast_today.get(k) for k in ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'precipitation_probability_max', 'weather_code'])
        weather_desc_str = interpret_weather_code(weather_desc)
        advice.append(f"**Today ({datetime.strptime(daily_forecast_today['time'], '%Y-%m-%d').strftime('%d %b')} - {weather_desc_str}):** Max: {today_max}°C, Min: {today_min}°C, Precip: {precip_sum}mm ({precip_prob}%)")
        if today_max and today_max > 35: advice.append("☀️ Heat Stress: Ensure water, shade. Reduce handling."); notifications.append("🔥 High temps today. Risk of heat stress.")
        if today_min and today_min < 10: advice.append("❄️ Cold Stress: Protect vulnerable. Dry bedding, extra feed."); notifications.append("🥶 Low temps. Protect vulnerable cattle.")
        if (precip_sum and precip_sum > 10) or (precip_prob and precip_prob > 60): advice.append("🌧️ Rain: Ensure shelter. Check mud. Secure feed.");
        if precip_sum and precip_sum > 25 : notifications.append(f"💧 Heavy rain ({precip_sum}mm). Risk of local flooding.")
    if daily_forecast_next_days: # Simplified checks for brevity
        if sum(1 for d in daily_forecast_next_days[1:4] if d.get('temperature_2m_max',0)>38) >=2 : notifications.append("♨️ Extended Heat Wave expected.")
        if sum(1 for d in daily_forecast_next_days[1:4] if d.get('temperature_2m_min',30)<5) >=2 : notifications.append("🧊 Extended Cold Spell expected.")
    return advice, notifications

def generate_animal_vaccination_report_pdf(animal_info, vaccination_records):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Vaccination Report for: {animal_info.get('name', 'N/A')} (Tag: {animal_info.get('tag_id', 'N/A')})", styles['h1']))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Breed:</b> {animal_info.get('breed', 'N/A')} | <b>Sex:</b> {animal_info.get('sex', 'N/A')} | <b>DOB:</b> {animal_info.get('dob', 'N/A')}", styles['Normal']))
    story.append(Spacer(1, 12))

    if not vaccination_records:
        story.append(Paragraph("No vaccination records found for this animal.", styles['Normal']))
    else:
        story.append(Paragraph("<b>Vaccination History:</b>", styles['h3']))
        data = [["Vaccine Name", "Date Administered", "Next Due", "Batch No.", "Admin By", "Notes"]]
        for record in vaccination_records:
             data.append([
                record[0] or "N/A", record[1] or "N/A", record[2] or "N/A",
                record[3] or "N/A", record[4] or "N/A", record[5] or "N/A"
            ])

        table = Table(data, colWidths=[doc.width/6.0]*6)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4CAF50")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#E8F5E9")),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('WORDWRAP', (0,0), (-1,-1), 'CJK')
        ]))
        story.append(table)
    story.append(Spacer(1, 24))
    story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}", styles['Normal'])) # CORRECTED
    story.append(Paragraph("Disclaimer: This report is based on data entered by the user.", styles['Normal'])) # CORRECTED
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_animal_vaccination_report_excel(animal_info, vaccination_records):
    output = BytesIO()
    df_data = []
    for record in vaccination_records:
        df_data.append({
            "Vaccine Name": record[0] or "N/A",
            "Date Administered": record[1] or "N/A",
            "Next Due Date": record[2] or "N/A",
            "Batch Number": record[3] or "N/A",
            "Administered By": record[4] or "N/A",
            "Notes": record[5] or "N/A"
        })
    df = pd.DataFrame(df_data)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=f"Vaccinations_{animal_info.get('tag_id','animal')}", index=False)
    return output.getvalue()

def generate_herd_details_excel(herd_data_tuples, username):
    output = BytesIO()
    df_herd = pd.DataFrame(herd_data_tuples, columns=["DB ID", "Tag ID", "Name", "Breed", "Sex", "DOB", "Status", "Lactation#", "Pregnancy", "EDD"])
    for col in ["DOB", "EDD"]:
        if col in df_herd.columns:
            df_herd[col] = pd.to_datetime(df_herd[col], errors='coerce').dt.strftime('%Y-%m-%d')
    def calculate_age_for_excel(dob_str):
        if pd.isna(dob_str) or not dob_str: return "N/A"
        try:
            birth_date = datetime.strptime(dob_str, '%Y-%m-%d').date()
            age_delta = date.today() - birth_date
            years = age_delta.days // 365
            months = (age_delta.days % 365) // 30
            return f"{years}y {months}m"
        except: return "N/A"
    df_herd['Age'] = df_herd['DOB'].apply(calculate_age_for_excel)
    excel_columns_ordered = ["Tag ID", "Name", "Breed", "Sex", "DOB", "Age", "Status", "Lactation#", "Pregnancy", "EDD", "DB ID"]
    df_herd = df_herd[excel_columns_ordered]
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_herd.to_excel(writer, sheet_name=f"{username}_Herd_Details", index=False)
    return output.getvalue()

# Helper function (if not already defined elsewhere in your app1.py)
def calf_rearing_table_exists(cursor):
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='calf_rearing_log';")
        return cursor.fetchone() is not None
    except sqlite3.Error as e:
        logger.error(f"Error checking calf_rearing_log table: {e}")
        return False
    
# Eco Practices Page (All the new text variables)
ECO_PAGE_TITLE_ENGLISH = "🌱 Eco-Friendly & Sustainable Farming Practices"
ECO_PAGE_DESC_ENGLISH = """
    Adopt practices that benefit the environment, improve soil health, conserve resources, and enhance long-term farm resilience.
"""

# Organic Farming
ORG_FARMING_SUBHEADER_ENGLISH = "🌿 Organic Farming"
ORG_FARMING_DETAILS_TITLE_ENGLISH = "Details on Organic Farming"
ORG_FARMING_DESC_ENGLISH = """
    **Description:** Avoids synthetic fertilizers, pesticides, GMOs. Relies on natural inputs and processes.
    **Benefits:**
    - Improves soil health and biodiversity.
    - Reduces water pollution from chemical runoff.
    - Produces potentially healthier food (residue-free).
    - Can fetch premium prices for certified produce.

    **Implementation:**
    - Use compost, manure, green manures for fertility.
    - Employ crop rotation, companion planting, biological pest control.
    - Source organic seeds/inputs.
    - Maintain buffer zones from conventional farms.
    - Certification process required for 'Organic' label (can be complex/costly).

    **Challenges:**
    - Potentially lower yields initially
    - Higher labor input
    - Weed and pest control can be difficult.
"""

# Crop Rotation
CROP_ROTATION_SUBHEADER_ENGLISH = "🔄 Crop Rotation"
CROP_ROTATION_DETAILS_TITLE_ENGLISH = "Details on Crop Rotation"
CROP_ROTATION_DESC_ENGLISH = """
    **Description:** Systematically changing the type of crop grown on a piece of land each season or year.
    **Benefits:**
    - Improves soil structure and fertility (e.g., legumes fix nitrogen).
    - Breaks pest and disease cycles specific to certain crops.
    - Suppresses weeds by alternating competitive crops.
    - Distributes nutrient uptake from different soil depths.

    **Implementation:**
    - Plan rotation sequences considering crop families (avoid planting related crops consecutively).
    - Include deep-rooted and shallow-rooted crops.
    - Incorporate legume cover crops.
    - Consider market demand and crop suitability.

    **Challenges:**
    - Requires careful planning
    - Market fluctuations for different crops.
"""

# Water Conservation
WATER_CONS_SUBHEADER_ENGLISH = "💧 Water Conservation"
WATER_CONS_DETAILS_TITLE_ENGLISH = "Details on Water Conservation"
WATER_CONS_DESC_ENGLISH = """
    **Description:** Using water resources efficiently in agriculture.
    **Benefits:**
    - Saves a critical resource, especially in water-scarce regions.
    - Reduces energy costs for pumping.
    - Minimizes soil erosion and nutrient leaching.
    - Can improve crop yields by providing water directly to roots.

    **Implementation:**
    - **Drip Irrigation:** Delivers water directly to the root zone.
    - **Sprinkler Systems:** More efficient than flood irrigation.
    - **Rainwater Harvesting:** Collect and store rainwater in ponds or tanks.
    - **Mulching:** Covering soil (organic or plastic) reduces evaporation.
    - **Laser Land Leveling:** Creates uniform slope for efficient surface irrigation.
    - **Contour Farming/Bunds:** Slows water runoff on slopes.

    **Challenges:**
    - Initial investment cost for systems like drip irrigation
    - Requires maintenance.
"""

# Integrated Pest Management
IPM_SUBHEADER_ENGLISH = "🐞 Integrated Pest Management"
IPM_DETAILS_TITLE_ENGLISH = "Details on IPM"
IPM_DESC_ENGLISH = """
    **Description:** Holistic approach using multiple tactics to control pests, minimizing reliance on chemical pesticides.
    **Benefits:**
    - Reduces pesticide use and environmental contamination.
    - Protects beneficial insects (pollinators, predators).
    - Lowers risk of pesticide resistance.
    - Can be more cost-effective long-term.

    **Implementation:**
    - **Monitoring:** Regularly scout fields to identify pests and assess damage levels.
    - **Cultural Controls:** Crop rotation, resistant varieties, sanitation.
    - **Biological Controls:** Introduce or encourage natural enemies (predators, parasitoids).
    - **Physical/Mechanical Controls:** Traps, barriers, hand-picking.
    - **Chemical Controls:** Use targeted, least-toxic pesticides only when necessary (based on thresholds).

    **Challenges:**
    - Requires knowledge of pest lifecycles and natural enemies
    - Can be more complex than simple spraying.
"""

# Manure Management
MANURE_MGMT_SUBHEADER_ENGLISH = "💩 Manure Management"
MANURE_MGMT_DETAILS_TITLE_ENGLISH = "Details on Manure Management"
MANURE_MGMT_DESC_ENGLISH = """
    **Description:** Proper handling, storage, and application of animal manure to utilize nutrients and prevent pollution.
    **Benefits:**
    - Recycles valuable nutrients (N, P, K) back to the soil.
    - Improves soil organic matter and structure.
    - Reduces reliance on synthetic fertilizers.
    - Prevents water contamination from runoff.
    - Potential for biogas generation.

    **Implementation:**
    - **Collection:** Regular collection from sheds/pens.
    - **Storage:** Covered storage (pits or heaps) to prevent nutrient loss and runoff.
    - **Composting:** Speeds up decomposition, reduces pathogens, stabilizes nutrients.
    - **Application:** Apply based on soil tests and crop needs, incorporate into soil quickly.
    - Avoid applying near water bodies or during heavy rain.

    **Challenges:**
    - Requires labor and space for handling/storage
    - Odor management
    - Pathogen risks if not composted properly.
"""

# Vermicomposting
VERMICOMPOSTING_SUBHEADER_ENGLISH = "🪱 Vermicomposting"
VERMICOMPOSTING_DETAILS_TITLE_ENGLISH = "Details on Vermicomposting"
VERMICOMPOSTING_DESC_ENGLISH = """
    **Description:** Using earthworms (like Eisenia fetida) to decompose organic waste into high-quality compost (vermicast).
    **Benefits:**
    - Produces nutrient-rich organic fertilizer quickly.
    - Improves soil aeration, water retention, and microbial activity.
    - Diverts organic waste from landfills/burning.
    - Vermicast can suppress some soil-borne diseases.

    **Implementation:**
    - Use suitable bins or pits with drainage.
    - Maintain optimal moisture (~70%) and temperature (15-25°C).
    - Feed worms appropriate organic matter (cow dung, crop residues, kitchen waste - avoid oily/meat).
    - Harvest vermicast periodically.

    **Challenges:**
    - Requires specific worm species
    - Sensitive to temperature and moisture extremes
    - Needs regular management.
"""

# Biogas Production
BIOGAS_SUBHEADER_ENGLISH = "🔥 Biogas Production"
BIOGAS_DETAILS_TITLE_ENGLISH = "Details on Biogas Production"
BIOGAS_DESC_ENGLISH = """
    **Description:** Anaerobic digestion of organic matter (mainly cow dung) to produce methane gas for fuel and nutrient-rich slurry.
    **Benefits:**
    - Provides clean, renewable cooking fuel, reducing reliance on firewood/LPG.
    - Produces high-quality organic fertilizer (slurry).
    - Improves sanitation by managing waste.
    - Reduces greenhouse gas emissions (methane capture).

    **Implementation:**
    - Construct a biogas digester (fixed dome or floating drum type).
    - Feed daily with a mixture of dung and water.
    - Use the produced gas for cooking/lighting via pipes.
    - Utilize the slurry as fertilizer after storage.

    **Challenges:**
    - Initial construction cost
    - Requires consistent supply of dung/water
    - Temperature affects gas production.
"""

# Agroforestry
AGROFORESTRY_SUBHEADER_ENGLISH = "🌳 Agroforestry"
AGROFORESTRY_DETAILS_TITLE_ENGLISH = "Details on Agroforestry"
AGROFORESTRY_DESC_ENGLISH = """
    **Description:** Integrating trees and shrubs with crops and/or livestock on the same land.
    **Benefits:**
    - Diversifies farm income (timber, fruit, fodder).
    - Improves soil health (nutrient cycling, erosion control).
    - Enhances biodiversity (habitat for birds, insects).
    - Provides shade for livestock, acts as windbreak.
    - Sequester carbon.

    **Implementation:**
    - Choose suitable tree species compatible with crops/livestock.
    - Design spatial arrangement (alley cropping, boundary planting, silvopasture).
    - Manage trees (pruning, thinning) to minimize competition with crops.

    **Challenges:**
    - Competition for light, water, nutrients between trees and crops
    - Longer time frame for returns from trees.
"""

# Rotational Grazing
ROTATIONAL_GRAZING_SUBHEADER_ENGLISH = "🌱 Rotational Grazing"
ROTATIONAL_GRAZING_DETAILS_TITLE_ENGLISH = "Details on Rotational Grazing"
ROTATIONAL_GRAZING_DESC_ENGLISH = """
    **Description:** A livestock management strategy that involves dividing pasture into sections and rotating grazing areas to optimize grass growth and soil health.
    **Benefits:**
    - Prevents overgrazing and allows vegetation to recover.
    - Improves soil fertility by evenly distributing manure.
    - Enhances pasture biodiversity and forage quality.
    - Reduces erosion and maintains healthy ground cover.

    **Implementation:**
    - Divide pasture into multiple paddocks or sections.
    - Rotate livestock between paddocks based on grass growth and recovery rates.
    - Provide access to clean water in each grazing area.
    - Monitor pasture health regularly to adjust grazing schedules.

    **Challenges:**
    - Initial setup can be resource-intensive (fences, water systems).
    - Requires regular monitoring and management of livestock.
    - May need supplemental feed during pasture recovery periods.
"""

# Cow Products Section
COW_PRODUCTS_TITLE_ENGLISH = "🐄 Indigenous Cow Products & Panchagavya Significance"
COW_PRODUCTS_DESC_ENGLISH = """
    The indigenous Indian cow (Bos Indicus) is revered not just for its milk but for the holistic benefits
    derived from all its by-products, collectively often referred to through the concept of Panchagavya.
    These products have traditional, agricultural, medicinal, and household applications.
"""

# Milk
MILK_SUBHEADER_ENGLISH = "🥛 Milk (Dugdha)"
MILK_DETAILS_TITLE_ENGLISH = "Details on Milk"
MILK_DESC_ENGLISH = """
    - **Nutritional Powerhouse:** Rich in calcium, protein, vitamins (A, D, B-complex), and easily digestible fats.
    - **A2 Beta-Casein:** Milk from many indigenous Indian breeds is predominantly A2, which some studies suggest is easier to digest and may have health benefits compared to A1 milk.
    - **Ayurvedic Importance:** Considered 'Sattvic' (pure), promoting ojas (vitality), and used as a base for many herbal medicines.
    - **Uses:** Direct consumption, curd, buttermilk, butter, ghee, paneer, khoa, sweets.
"""

# Ghee
GHEE_SUBHEADER_ENGLISH = "🧈 Ghee (Ghrita)"
GHEE_DETAILS_TITLE_ENGLISH = "Details on Ghee"
GHEE_DESC_ENGLISH = """
    - **Clarified Butter:** Made by simmering butter, removing milk solids and water.
    - **Ayurvedic Significance:** Highly prized; used for cooking, medicinal preparations (as a carrier for herbs - 'anupana'), and in religious rituals (agnihotra, lamps). Believed to enhance digestion, memory, and immunity.
    - **Nutritional Aspects:** Rich in healthy fats, fat-soluble vitamins (A, D, E, K). High smoke point makes it good for cooking.
    - **Uses:** Culinary, Ayurvedic medicine, religious ceremonies, skincare.
"""

# Dung
DUNG_SUBHEADER_ENGLISH = "💩 Dung (Gomaya)"
DUNG_DETAILS_TITLE_ENGLISH = "Details on Cow Dung"
DUNG_DESC_ENGLISH = """
    - **Organic Manure:** Excellent natural fertilizer, improves soil structure, water retention, and microbial activity. Key component of Jaivik Krishi (organic farming).
        - *Jeevamrutha & Beejamrutha:* Fermented concoctions using dung, urine, jaggery, and gram flour to enrich soil and treat seeds.
    - **Bio-Fuel:** Dried dung cakes are a traditional cooking fuel in rural India. Also used in biogas plants to produce methane gas for cooking and lighting, with the slurry being a rich fertilizer.
    - **Traditional Plaster:** Mixed with mud for plastering walls and floors (provides insulation, insect-repellent properties).
    - **Pest Repellent:** Smoke from burning dung can repel insects. Some preparations are used as bio-pesticides.
    - **Cleansing Agent:** Traditionally used for cleaning floors and courtyards due to perceived antimicrobial properties.
    - **Other Products:** Dhoop (incense sticks), mosquito coils, crafting material for idols/decorative items.
"""

# Curd/Yogurt
CURD_SUBHEADER_ENGLISH = "🥣 Curd/Yogurt (Dahi)"
CURD_DETAILS_TITLE_ENGLISH = "Details on Curd"
CURD_DESC_ENGLISH = """
    - **Probiotic Rich:** Contains beneficial bacteria that aid digestion and boost gut health.
    - **Cooling Properties:** Considered cooling for the body in Ayurveda.
    - **Nutritional Value:** Good source of calcium, protein, and B vitamins.
    - **Uses:** Direct consumption, Lassi, Raita, Buttermilk (Chaas), Kadhi, marinades.
"""

# Urine
URINE_SUBHEADER_ENGLISH = "💧 Urine (Gomutra)"
URINE_DETAILS_TITLE_ENGLISH = "Details on Cow Urine"
URINE_DESC_ENGLISH = """
    - **Traditional Medicine (Ayurveda):** Used in various formulations (e.g., Gomutra Ark - distilled cow urine) for perceived therapeutic properties. Claims include detoxification, antimicrobial effects, and boosting immunity. *Scientific validation for many human medicinal claims is ongoing and requires rigorous research.*
    - **Agricultural Uses:**
        - *Bio-pesticide/Insecticide:* Fermented or diluted cow urine is sprayed on crops to repel pests.
        - *Growth Promoter:* Some farmers use it as a liquid fertilizer due to its nitrogen content.
        - *Seed Treatment:* Used to treat seeds before sowing.
    - **Cleansing Agent:** Traditionally used as a disinfectant for floors and in some purification rituals.
    - **Modern Products:** Soaps, floor cleaners, and other consumer goods incorporating cow urine are being marketed.
"""
URINE_NOTE_ENGLISH = "Note: While Gomutra holds significance in traditional practices, consult with qualified Ayurvedic practitioners for medicinal uses and rely on scientific evidence for health claims."

# Panchagavya
PANCHAGAVYA_SUBHEADER_ENGLISH = "🌿 Panchagavya - The Concoction"
PANCHAGAVYA_DETAILS_TITLE_ENGLISH = "Details on Panchagavya"
PANCHAGAVYA_DESC_ENGLISH = """
    - **Definition:** A traditional Ayurvedic formulation made by mixing five components of indigenous cow products: cow dung, cow urine, milk, curd, and ghee, often with other ingredients like water, jaggery, and banana.
    - **Agricultural Uses:**
        - **Bio-fertilizer:** Acts as a powerful organic fertilizer, improving soil fertility, plant growth, and yield.
        - **Pest & Disease Control:** Helps in managing various plant pests and diseases naturally.
        - **Plant Growth Regulator:** Promotes flowering, fruiting, and overall plant vigor.
    - **Medicinal Uses:** Traditionally used in Ayurveda for various ailments, believed to boost immunity, aid digestion, and have antimicrobial properties. *Further scientific research is needed to validate all medicinal claims.*
    - **Benefits:**
        - Enhances soil microbial activity.
        - Reduces dependency on synthetic chemicals.
        - Environmentally friendly and sustainable.
        - Improves crop quality and shelf life.
"""

# Other Value-Added Products
OTHER_PRODUCTS_SUBHEADER_ENGLISH = "🌿 Other Value-Added Products from Indigenous Cows"
OTHER_PRODUCTS_DESC_ENGLISH = """
    Beyond the core Panchagavya, farmers can create various marketable products:
    - **Vermicompost:** High-quality organic fertilizer produced by earthworms feeding on cow dung and agricultural waste.
    - **Dhoop/Agarbatti (Incense):** Made from dried cow dung mixed with herbs and natural binders.
    - **Dung Pots/Bricks:** Eco-friendly alternatives for planters or construction.
    - **Bio-Pesticides & Growth Promoters:** Formulations like Jeevamrutha, Beejamrutha, Agnihastra.
    - **Traditional Sweets & Dairy Delicacies:** Unique products made from indigenous cow milk.
"""

# Calculator Section
TOOLS_HEADER_ENGLISH = "🛠️ Tools for Sustainability Assessment"
CARBON_ESTIMATOR_TITLE_ENGLISH = "🌍 Carbon Footprint Estimator"
CARBON_ESTIMATOR_DESC_ENGLISH = "Estimate your farm's approximate **monthly** carbon emissions."
FUEL_USAGE_LABEL_ENGLISH = "Diesel/Petrol Usage (Liters/month):"
FERTILIZER_USAGE_LABEL_ENGLISH = "Nitrogen Fertilizer Usage (Kg N/month):"
LIVESTOCK_COUNT_LABEL_ENGLISH = "Number of Adult Cattle:"
RICE_PADDY_AREA_LABEL_ENGLISH = "Area under Rice Paddy (Acres, if applicable):"
ESTIMATE_FOOTPRINT_BUTTON_ENGLISH = "Estimate Footprint"
ESTIMATED_FOOTPRINT_SUCCESS_ENGLISH = "Estimated Monthly Footprint: ~{total_emissions:.1f} kg CO₂e"
CARBON_NOTE_ENGLISH = "Note: This is a rough estimate based on general factors."

WATER_CALCULATOR_TITLE_ENGLISH = "💧 Water Usage Calculator"
WATER_CALCULATOR_DESC_ENGLISH = "Estimate monthly water usage for irrigation."
FIELD_SIZE_LABEL_ENGLISH = "Irrigated Field Size (Acres):"
IRRIGATION_DEPTH_LABEL_ENGLISH = "Avg. Daily Irrigation Depth per Acre (mm):"
DAYS_IRRIGATED_LABEL_ENGLISH = "Number of Irrigation Days per Month:"
ESTIMATE_WATER_USAGE_BUTTON_ENGLISH = "Estimate Water Usage"
ESTIMATED_WATER_USAGE_SUCCESS_ENGLISH = "Estimated Monthly Water Usage: {monthly_water_usage:,.0f} Liters"
WATER_NOTE_ENGLISH = "(Based on {irrigation_per_acre} mm/day application)"

INFO_MESSAGE_ENGLISH = "This page aims to provide general information. For specific applications or health-related uses, professional consultation is always recommended."

# --- VIDEO MAPPING (NO INDENTATION) ---
# Define video URLs for different practices.
# You can use YouTube embed links or direct MP4 links.
# For uneducated farmers, short, clear, visual demonstrations are key.
VIDEO_URLS = {
    "🌿 Organic Farming": "https://www.youtube.com/embed/eKC7QLepYKA?si=FITJLlXr7qmKcDe0",
    "🔄 Crop Rotation": "https://www.youtube.com/embed/64YapgN6G2w",
    "💧 Water Conservation": "https://www.youtube.com/embed/j0yjZdCPOSQ",
    "🐞 Integrated Pest Management": "https://www.youtube.com/embed/ABVvq6i8MF4",
    "💩 Manure Management": "https://www.youtube.com/embed/NZ8Q7SmJQYE",
    "🪱 Vermicomposting": "https://www.youtube.com/embed/AF-jzWKMdwE",
    "🔥 Biogas Production": "https://www.youtube.com/embed/3UafRz3QeO8",
    "🌳 Agroforestry": "https://www.youtube.com/embed/jLZ0KtNx354",
    "🌱 Rotational Grazing": "https://www.youtube.com/embed/ljIzUcSwJxM",
    "🌿 Panchagavya - The Concoction": "https://www.youtube.com/embed/dgHtvp6X3ms",
    "Drip Irrigation": "https://www.youtube.com/embed/your-drip-irrigation-video-id"
}


# --- User Authentication Functions ---
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
def verify_password(plain_password, hashed_password_from_db):
    if isinstance(hashed_password_from_db, str): # Ensure it's a string before encoding
        hashed_password_from_db = hashed_password_from_db.encode('utf-8')
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password_from_db)

if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'user_id' not in st.session_state: st.session_state.user_id = None
if 'username' not in st.session_state: st.session_state.username = None
if 'role' not in st.session_state: st.session_state.role = None
if 'current_page' not in st.session_state: st.session_state.current_page = "Home"

# --- At the top with other menu option lists ---
public_options = ["Home", "Breed Info", "Farm Products", "Eco Practices", "Temple Connect", "Identify Breed", "Chatbot", "Price Trends", "Diagnosis", "Govt Schemes", "Lifecycle Management", "Donate & Adopt", "Community Network", "Login", "Register"]
public_icons = ["house-gear-fill", "info-square-fill", "shop", "recycle", "house-heart", "camera-fill", "chat-quote-fill", "graph-up-arrow", "clipboard2-pulse-fill", "bank", "arrow-repeat", "heart-half", "chat-left-dots-fill", "box-arrow-in-right", "person-plus-fill"]

farmer_options = ["Farmer Dashboard", "Breeding", "My Herd", "Sell Cattle", "Farm Products", "Eco Practices", "Identify Breed", "Vet Locator", "Chatbot", "Browse Machinery", "Sell Machinery", "Nutrition Planner", "Fair Price Guide", "Price Trends", "Diagnosis", "Govt Schemes", "Lifecycle Management", "Community Network", "My Profile", "Logout"]
farmer_icons = ["speedometer2", "heart-pulse-fill", "grid-fill", "cart-plus-fill", "shop", "recycle", "camera-fill", "geo-alt", "chat-quote-fill", "search", "tools", "calculator-fill", "cash-stack", "graph-up-arrow", "clipboard2-pulse-fill", "bank", "arrow-repeat", "chat-left-dots-fill", "person-circle", "box-arrow-right"]

buyer_options = ["Buyer Dashboard", "Browse Cattle", "Farm Products", "Fair Price Guide", "Eco Practices", "Chatbot", "Browse Machinery", "Sell Machinery", "Saved Alerts", "Price Trends", "Govt Schemes", "Lifecycle Management", "Donate & Adopt", "Community Network", "Logout"]
buyer_icons = ["speedometer2", "search", "shop", "cash-stack", "recycle", "chat-quote-fill", "tools", "cash-coin", "bell-fill", "graph-up-arrow", "bank", "arrow-repeat", "heart-half", "chat-left-dots-fill", "box-arrow-right"]

# Add to PAGES_REQUIRING_LOGIN if you want them protected, or keep public
# For now, let's make them public for wider accessibility.
# PAGES_REQUIRING_LOGIN = [..., "Find a Farmer", "Vet Locator"] # If login required
if st.session_state.logged_in:
    if st.session_state.role == "farmer":
        options_to_show = farmer_options
        icons_to_show = farmer_icons
        default_page_after_login = "Farmer Dashboard" # CHANGED
    elif st.session_state.role == "buyer":
        options_to_show = buyer_options
        icons_to_show = buyer_icons
        default_page_after_login = "Buyer Dashboard"
    else:
        options_to_show = public_options
        icons_to_show = public_icons
        default_page_after_login = "Home"
    try:
        default_index = options_to_show.index(st.session_state.current_page) if st.session_state.current_page in options_to_show else options_to_show.index(default_page_after_login)
    except ValueError: default_index = 0
else:
    options_to_show = public_options
    icons_to_show = public_icons
    try:
        default_index = options_to_show.index(st.session_state.current_page) if st.session_state.current_page in options_to_show else 0
    except ValueError: default_index = 0
if default_index >= len(options_to_show): default_index = 0

selected_page = option_menu(
    menu_title=None, options=options_to_show, icons=icons_to_show, menu_icon="cow",
    default_index=default_index, orientation="horizontal",
    styles={ # Your existing styles
        "container": {"padding": "2px 8px", "background-color": "#e8f5e9", "border-radius": "6px"},
        "icon": {"color": "#1e8449", "font-size": "14px"},
        "nav-link": {"font-size": "11px", "font-weight": "500", "color": "#000000", "text-align": "center", "margin": "0px 3px", "--hover-color": "#c8e6c9", "padding": "5px 7px"},
        "nav-link-selected": {"background-color": "#2e7d32", "color": "#ffffff", "font-weight": "600"},
    }
)
st.session_state.current_page = selected_page

# Ensure these are initialized at the TOP of your app1.py
if 'expanded_cattle_listing_id' not in st.session_state:
    st.session_state.expanded_cattle_listing_id = None
if 'expanded_machinery_listing_id' not in st.session_state:
    st.session_state.expanded_machinery_listing_id = None

# --- PAGE CONTENT ROUTING ---

PAGES_REQUIRING_LOGIN = ["Farmer Dashboard", "My Herd", "Sell Cattle", "Browse Machinery", "Sell Machinery", "Nutrition Planner", "Breeding", "Browse Cattle", "Saved Alerts","My Profile"]
PAGES_REQUIRING_FARMER_ROLE = ["Farmer Dashboard", "My Herd", "Sell Cattle", "Nutrition Planner","Breeding","My Profile"] # Farmer specific actions. "Sell Machinery" and "Browse Machinery" can be open to logged-in users. "Breeding" can also be farmer-centric.
PAGES_REQUIRING_BUYER_ROLE = ["Buyer Dashboard","Browse Cattle", "Saved Alerts"] # Buyer specific actions.
show_page_content = True
if selected_page in PAGES_REQUIRING_LOGIN and not st.session_state.logged_in:
    st.warning(f"You need to be logged in to access the '{selected_page}' page.")
    if st.button("Login to Continue", key=f"login_redirect_{selected_page}"):
        st.session_state.current_page = "Login"
        st.rerun()
    show_page_content = False
elif st.session_state.logged_in: # Only check role if logged in
    if selected_page in PAGES_REQUIRING_FARMER_ROLE and st.session_state.role != 'farmer':
        st.warning(f"You need to be logged in as a Farmer to access '{selected_page}'. You are logged in as a {st.session_state.role}.")
        show_page_content = False
    elif selected_page in PAGES_REQUIRING_BUYER_ROLE and st.session_state.role != 'buyer':
        st.warning(f"You need to be logged in as a Buyer to access '{selected_page}'. You are logged in as a {st.session_state.role}.")
        show_page_content = False
        


# 0. Login Page
if selected_page == "Login":
    st.title("👤 Login to Kamadhenu Program")
    if st.session_state.logged_in:
        st.success(f"You are already logged in as {st.session_state.username} ({st.session_state.role})!")
        default_redirect = "My Herd" if st.session_state.role == "farmer" else "Browse Cattle"
        if st.button(f"Go to {default_redirect}", key="auto_btn_2"):
            st.session_state.current_page = default_redirect; st.rerun()
    else:
        with st.form("login_form"):
            login_username = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if not login_username or not login_password: st.error("Please enter both username and password.")
                else:
                    conn = get_connection()
                    if conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT user_id, username, password_hash, role FROM users WHERE username = ?", (login_username,))
                        user_data = cursor.fetchone()
                        if user_data and verify_password(login_password, user_data[2]):
                            st.session_state.logged_in = True
                            st.session_state.user_id = user_data[0]
                            st.session_state.username = user_data[1]
                            st.session_state.role = user_data[3] # Store role
                            # --- MODIFIED REDIRECT ---
                            if user_data[3] == "farmer":
                                st.session_state.current_page = "Farmer Dashboard"
                            elif user_data[3] == "buyer":
                                st.session_state.current_page = "Buyer Dashboard" # <<< CHANGE HERE
                            else:
                                st.session_state.current_page = "Home" # Fallback if role is unexpected
                            # --- END MODIFIED REDIRECT ---
                            logger.info(f"User {st.session_state.username} ({st.session_state.role}) logged in. Redirecting to {st.session_state.current_page}")
                            st.success(f"Welcome back, {st.session_state.username}!");
                            st.rerun()          
                        else: st.error("Invalid username or password.")
                    else: st.error("Database connection failed.")
        st.markdown("---"); st.write("Don't have an account?")
        if st.button("Register Here", key="auto_btn_3"): st.session_state.current_page = "Register"; st.rerun()

# 0. Registration Page
elif selected_page == "Register":
    st.title(translate_text("📝 Register for Kamadhenu Program", current_lang))
    if st.session_state.logged_in: 
        st.success(translate_text(f"Already logged in as {st.session_state.username}!", current_lang))
    else:
        with st.form("registration_form"):
            reg_username = st.text_input(translate_text("Choose a Username*", current_lang))
            reg_password = st.text_input(translate_text("Choose a Password*", current_lang), type="password")
            reg_confirm_password = st.text_input(translate_text("Confirm Password*", current_lang), type="password")
            
            # Role selection during registration
            reg_role = st.radio(translate_text("I am a:*", current_lang), 
                                ('Farmer', 'Buyer'), 
                                horizontal=True, 
                                key="reg_role_select")
            
            reg_email = st.text_input(translate_text("Email (Optional)", current_lang))
            reg_full_name = st.text_input(translate_text("Full Name (Optional)", current_lang))
            reg_region_user = st.text_input(translate_text("Your Region/State (Optional)", current_lang)) 

            # NEW: UPI ID input during registration
            reg_upi_id = st.text_input(
                translate_text("Your UPI ID (Optional)", current_lang),
                help=translate_text("If you're a Farmer, providing your UPI ID allows buyers to pay you directly.", current_lang)
            )

            reg_share_contact = st.checkbox(
                translate_text("Allow other users to see my email/phone for listings I post?", current_lang), 
                value=False, 
                key="reg_share"
            )
            
            submitted = st.form_submit_button(translate_text("Register", current_lang))
            
            if submitted:
                if not reg_username or not reg_password or not reg_confirm_password: 
                    st.error(translate_text("Username & Passwords required.", current_lang))
                elif reg_password != reg_confirm_password: 
                    st.error(translate_text("Passwords do not match.", current_lang))
                elif len(reg_password) < 6: 
                    st.error(translate_text("Password must be at least 6 characters.", current_lang))
                else:
                    conn = get_connection()
                    if conn:
                        cursor = conn.cursor()
                        try:
                            hashed_pw = hash_password(reg_password)
                            
                            # Insert with role (lowercase), email, full_name, region, share_contact_info, and upi_id
                            cursor.execute(
                                "INSERT INTO users (username, password_hash, role, email, full_name, region, share_contact_info, upi_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                (reg_username, hashed_pw, reg_role.lower(), 
                                 reg_email or None, reg_full_name or None, reg_region_user or None,
                                 1 if reg_share_contact else 0, reg_upi_id or None) # Added share_contact_info and upi_id
                            )
                            conn.commit()
                            st.success(translate_text("Registration successful! You can now log in.", current_lang))
                            logger.info(f"New user: {reg_username}, Role: {reg_role.lower()}")
                            st.session_state.current_page = "Login"
                            st.rerun()
                        except sqlite3.IntegrityError: 
                            st.error(translate_text("Username or Email already exists.", current_lang))
                        except Exception as e: 
                            st.error(translate_text(f"Registration error: {e}", current_lang))
                            logger.error(f"Reg error {reg_username}: {e}")
                    else: 
                        st.error(translate_text("Database connection failed.", current_lang))
        st.markdown("---")
        st.write(translate_text("Already have an account?", current_lang))
        if st.button(translate_text("Login Here", current_lang, key="auto_btn_4")): 
            st.session_state.current_page = "Login"
            st.rerun()
# 0. Logout
elif selected_page == "Logout":
    if st.session_state.logged_in:
        logger.info(f"User {st.session_state.username} logging out.")
        for key in ['logged_in', 'user_id', 'username', 'role']: # Clear all session keys related to login
            if key in st.session_state: del st.session_state[key]
        st.session_state.logged_in = False # Explicitly set
        st.session_state.current_page = "Home"; st.success("Logged out."); st.rerun()
    else: st.info("Not logged in.");
    if st.button("Go to Home", key="auto_btn_5"): st.session_state.current_page = "Home"; st.rerun()

# # CATTLE_BREEDS_DATA should be defined globally
# --- Start of "My Herd" Page ---
# Ensure these are defined globally or passed correctly:
# CATTLE_BREEDS_DATA, get_connection(), logger, datetime, date, timedelta, pd, sqlite3,
# calf_rearing_table_exists() (defined in my previous response)


elif selected_page == "My Herd" and st.session_state.logged_in and st.session_state.role == "farmer": # Added role check
    ts.title(f"🐄 My Herd - Welcome {st.session_state.username}!")
    ts.markdown("Manage your cattle records, health schedules, and production logs here.")
    st.markdown("---")

    conn = get_connection()
    if not conn: st.error(translate_text("Database connection failed. Please try again later.",current_lang)); st.stop()
    
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_cattle';")
        if cursor.fetchone() is None:
            st.error(translate_text("Critical Error: 'user_cattle' table not found. Please run database setup.",current_lang))
            st.stop()
    except sqlite3.Error as e:
        st.error(f"DB Error checking 'user_cattle' table: {e}"); st.stop()

    pre_selected_animal_id_from_reminders = st.session_state.pop('nav_to_my_herd_animal_id', None)
    pre_selected_tab_name_from_reminders = st.session_state.pop('nav_to_my_herd_tab_name', None)
    default_animal_selectbox_index_main = 0 # For the main animal selection dropdown
    default_tab_display_index = 0 # For tabs

    with st.expander(translate_text("➕ Add / Edit Animal Details & Breeding Info",current_lang), expanded=False):
        try:
            cursor.execute("SELECT cattle_id, tag_id, name FROM user_cattle WHERE user_id = ? ORDER BY name, tag_id", (st.session_state.user_id,))
            user_animals_for_edit = cursor.fetchall()
        except sqlite3.Error as e_fetch_animals:
            st.error(f"Error fetching animal list for editing: {e_fetch_animals}")
            user_animals_for_edit = []

        animal_options_for_edit = {f"{row[1] or 'No Tag'} - {row[2] or 'Unnamed'} (ID: {row[0]})": row[0] for row in user_animals_for_edit}
        animal_options_for_edit_list = ["Add New Animal"] + list(animal_options_for_edit.keys())
        
        current_default_animal_select_index = 0
        if pre_selected_animal_id_from_reminders:
            for i, option_display_name in enumerate(animal_options_for_edit_list):
                if option_display_name != "Add New Animal": # Skip the "Add New" option
                    if animal_options_for_edit.get(option_display_name) == pre_selected_animal_id_from_reminders:
                        current_default_animal_select_index = i
                        break
        
        selected_animal_to_edit_display = st.selectbox(
            "Select Animal to Edit or 'Add New Animal'",
            options=animal_options_for_edit_list,
            index=current_default_animal_select_index,
            key="herd_edit_animal_select_dropdown_v5"
        )
        editing_cattle_id_herd = None
        initial_data = {}
        if selected_animal_to_edit_display != "Add New Animal":
            editing_cattle_id_herd = animal_options_for_edit[selected_animal_to_edit_display]
            try:
                cursor.execute("""
                    SELECT tag_id, name, breed, sex, dob, purchase_date, purchase_price,
                           current_status, notes, sire_tag_id, dam_tag_id,
                           last_calving_date, last_heat_observed_date, last_insemination_date,
                           insemination_sire_tag_id, pregnancy_status, pregnancy_diagnosis_date,
                           expected_due_date, lactation_number
                    FROM user_cattle WHERE cattle_id = ? AND user_id = ?
                """, (editing_cattle_id_herd, st.session_state.user_id))
                data_row = cursor.fetchone() 
                if data_row:
                    initial_data = {
                        "tag_id": data_row[0] or "", "name": data_row[1] or "", "breed": data_row[2] or "", "sex": data_row[3],
                        "dob": datetime.strptime(data_row[4], '%Y-%m-%d').date() if data_row[4] else None,
                        "purchase_date": datetime.strptime(data_row[5], '%Y-%m-%d').date() if data_row[5] else None,
                        "purchase_price": data_row[6] if data_row[6] is not None else 0.0,
                        "current_status": data_row[7] or "Active", "notes": data_row[8] or "",
                        "sire_tag_id": data_row[9] or "", "dam_tag_id": data_row[10] or "",
                        "last_calving_date": data_row[11],"last_heat_observed_date": data_row[12],"last_insemination_date": data_row[13],
                        "insemination_sire_tag_id": data_row[14] or "","pregnancy_status": data_row[15],"pregnancy_diagnosis_date": data_row[16],
                        "expected_due_date": data_row[17],"lactation_number": data_row[18] if data_row[18] is not None else 0
                    }
            except sqlite3.Error as e_fetch_edit: st.error(f"Error fetching animal data for editing: {e_fetch_edit}")

        with st.form(translate_text("user_cattle_details_form_v5",current_lang), clear_on_submit=False):
            ts.subheader("📋 Animal Basic Details")
            h_c1, h_c2 = st.columns(2)
            tag_id_form_val = h_c1.text_input(translate_text("Ear Tag ID",current_lang), value=initial_data.get("tag_id", ""), key="form_tag_id_v5")
            name_form_val = h_c2.text_input(translate_text("Animal Name",current_lang), value=initial_data.get("name", ""), key="form_name_v5")
            h_c3, h_c4 = st.columns(2)
            breed_options_form_val = [translate_text("Select Breed",current_lang)] + [b['name'] for b in CATTLE_BREEDS_DATA] + ["Crossbred", "Other"]
            breed_idx_form_val = breed_options_form_val.index(initial_data.get("breed")) if initial_data.get("breed") in breed_options_form_val else 0
            breed_form_val = h_c3.selectbox(translate_text("Breed",current_lang), options=breed_options_form_val, index=breed_idx_form_val, key="form_breed_v5")
            sex_options_form_val = ['Female', 'Male', 'Cow', 'Bull', 'Heifer', 'Calf-Female', 'Calf-Male', 'Other', 'Not Specified']
            sex_initial_val = initial_data.get("sex", "Not Specified")
            sex_idx_form_val = sex_options_form_val.index(sex_initial_val) if sex_initial_val in sex_options_form_val else sex_options_form_val.index("Not Specified")
            sex_form_val = h_c4.selectbox(translate_text("Sex",current_lang), options=sex_options_form_val, index=sex_idx_form_val, key="form_sex_v5")
            h_c5, h_c6 = st.columns(2)
            dob_initial_val = initial_data.get("dob", date.today() - timedelta(days=730))
            dob_form_val = h_c5.date_input(translate_text("Date of Birth",current_lang), value=dob_initial_val, min_value=date(1980,1,1), max_value=date.today(), key="form_dob_v5")
            purchase_date_initial_val = initial_data.get("purchase_date")
            purchase_date_form_val = h_c6.date_input(translate_text("Purchase Date",current_lang), value=purchase_date_initial_val, max_value=date.today(), key="form_pdate_v5")
            h_c7, h_c8 = st.columns(2)
            purchase_price_form_val = h_c7.number_input(translate_text("Purchase Price (₹)",current_lang), min_value=0.0, value=initial_data.get("purchase_price", 0.0), step=100.0, key="form_pprice_v5")
            status_options_form_val = ['Active', 'For Sale', 'Sold', 'Deceased', 'Other']
            current_status_initial_val = initial_data.get("current_status", "Active")
            status_idx_form_val = status_options_form_val.index(current_status_initial_val) if current_status_initial_val in status_options_form_val else 0
            current_status_form_val = h_c8.selectbox(translate_text("Current Status",current_lang), options=status_options_form_val, index=status_idx_form_val, key="form_status_v5")
            h_c9, h_c10 = st.columns(2)
            sire_id_form_val = h_c9.text_input(translate_text("Sire Tag ID",current_lang), value=initial_data.get("sire_tag_id", ""), key="form_sire_id_v5")
            dam_id_form_val = h_c10.text_input(translate_text("Dam Tag ID",current_lang), value=initial_data.get("dam_tag_id", ""), key="form_dam_id_v5")
            notes_form_val = st.text_area(translate_text("General Notes",current_lang), value=initial_data.get("notes", ""), key="form_notes_v5")
            st.markdown("---"); ts.subheader("🐃 Breeding Cycle Information")
            is_female_for_breeding_form_val = sex_form_val in ['Female', 'Cow', 'Heifer']
            last_calving_date_input_f, last_heat_date_input_f, last_insem_date_input_f = None, None, None
            insem_sire_input_f = initial_data.get("insemination_sire_tag_id", "")
            preg_status_input_f = initial_data.get("pregnancy_status", "Not Applicable")
            preg_diag_date_input_f, edd_calculated_display_f = None, None
            lactation_number_input_f = initial_data.get("lactation_number", 0)
            if is_female_for_breeding_form_val:
                bc_fcol1, bc_fcol2 = st.columns(2)
                with bc_fcol1:
                    lcd_fval = initial_data.get("last_calving_date")
                    last_calving_date_input_f = st.date_input(translate_text("Last Calving Date",current_lang), value=datetime.strptime(lcd_fval, '%Y-%m-%d').date() if lcd_fval else None, max_value=date.today(), key="form_lcd_v5")
                    lhod_fval = initial_data.get("last_heat_observed_date")
                    last_heat_date_input_f = st.date_input(translate_text("Last Heat Observed",current_lang), value=datetime.strptime(lhod_fval, '%Y-%m-%d').date() if lhod_fval else None, max_value=date.today(), key="form_lhd_v5")
                    lid_fval = initial_data.get("last_insemination_date")
                    last_insem_date_input_f = st.date_input(translate_text("Last Insemination Date",current_lang), value=datetime.strptime(lid_fval, '%Y-%m-%d').date() if lid_fval else None, max_value=date.today(), key="form_lid_v5")
                with bc_fcol2:
                    insem_sire_input_f = st.text_input(translate_text("Sire/Semen Used",current_lang), value=initial_data.get("insemination_sire_tag_id", ""), key="form_insem_sire_v5")
                    preg_status_opts_f_val = ["Not Applicable", "Open", "Inseminated - Awaiting Check", "Confirmed Pregnant", "Due for Calving", "Recently Calved"]
                    ps_fval_init = initial_data.get("pregnancy_status", "Not Applicable")
                    ps_fidx_val = preg_status_opts_f_val.index(ps_fval_init) if ps_fval_init in preg_status_opts_f_val else 0
                    preg_status_input_f = st.selectbox(translate_text("Pregnancy Status",current_lang), options=preg_status_opts_f_val, index=ps_fidx_val, key="form_preg_status_v5")
                    pdd_fval = initial_data.get("pregnancy_diagnosis_date")
                    preg_diag_date_input_f = st.date_input(translate_text("Pregnancy Diagnosis Date",current_lang), value=datetime.strptime(pdd_fval, '%Y-%m-%d').date() if pdd_fval else None, max_value=date.today(), key="form_pdd_v5")
                lact_num_fval_init = initial_data.get("lactation_number", 0)
                lactation_number_input_f = st.number_input(translate_text("Lactation Number",current_lang), min_value=0, value=lact_num_fval_init, step=1, key="form_lact_num_v5")
                if last_insem_date_input_f and (preg_status_input_f == "Confirmed Pregnant" or preg_status_input_f == "Due for Calving"):
                    edd_calculated_display_f = last_insem_date_input_f + timedelta(days=283)
                    st.info(translate_text(f"Calculated EDD: {edd_calculated_display_f.strftime('%d %b, %Y')}",current_lang))
                elif initial_data.get("expected_due_date"):
                    try: edd_stored_disp = datetime.strptime(initial_data.get("expected_due_date"), '%Y-%m-%d').date(); st.info(translate_text(f"Stored EDD: {edd_stored_disp.strftime('%d %b, %Y')}",current_lang))
                    except: pass
            else: st.caption(translate_text("Breeding info for female animals.",current_lang))
            
            submit_cattle_details_btn = st.form_submit_button(translate_text("Save Animal Details",current_lang))
            if submit_cattle_details_btn:
                if not tag_id_form_val.strip() and not name_form_val.strip():
                    st.error(translate_text("Provide Tag ID or Animal Name.",current_lang))
                else:
                    final_breed_to_save_val = breed_form_val if breed_form_val != "Select Breed" else (initial_data.get("breed") or None)
                    db_lcd_val = last_calving_date_input_f if is_female_for_breeding_form_val else None
                    db_lhd_val = last_heat_date_input_f if is_female_for_breeding_form_val else None
                    db_lid_val = last_insem_date_input_f if is_female_for_breeding_form_val else None
                    db_insem_sire_val = insem_sire_input_f if is_female_for_breeding_form_val else None
                    db_preg_status_val = preg_status_input_f if is_female_for_breeding_form_val and preg_status_input_f != "Not Applicable" else None
                    db_pdd_val = preg_diag_date_input_f if is_female_for_breeding_form_val else None
                    db_edd_val_to_store = edd_calculated_display_f 
                    if not db_edd_val_to_store and initial_data.get("expected_due_date"):
                        try: db_edd_val_to_store = datetime.strptime(initial_data.get("expected_due_date"), '%Y-%m-%d').date()
                        except: db_edd_val_to_store = None
                    db_lact_num_val = lactation_number_input_f if is_female_for_breeding_form_val else 0
                    
                    # --- ADDED PRE-CHECKS FOR TAG ID UNIQUENESS ---
                    tag_id_to_check = tag_id_form_val.strip()
                    can_proceed = True
                    if tag_id_to_check: 
                        if editing_cattle_id_herd: 
                            cursor.execute("""SELECT 1 FROM user_cattle WHERE user_id = ? AND tag_id = ? AND cattle_id != ?""",
                                           (st.session_state.user_id, tag_id_to_check, editing_cattle_id_herd))
                            if cursor.fetchone():
                                st.error(f"Error: Tag ID '{tag_id_to_check}' already exists for another animal. Use a unique Tag ID.")
                                can_proceed = False
                        else: # Adding new animal
                            cursor.execute("""SELECT 1 FROM user_cattle WHERE user_id = ? AND tag_id = ?""",
                                           (st.session_state.user_id, tag_id_to_check))
                            if cursor.fetchone():
                                st.error(f"Error: Tag ID '{tag_id_to_check}' already exists. Use a unique Tag ID.")
                                can_proceed = False
                    # --- END PRE-CHECKS ---

                    if can_proceed:
                        try:
                            if editing_cattle_id_herd:
                                cursor.execute("""
                                    UPDATE user_cattle SET
                                    tag_id=?, name=?, breed=?, sex=?, dob=?, purchase_date=?, purchase_price=?, current_status=?, notes=?, sire_tag_id=?, dam_tag_id=?,
                                    last_calving_date=?, last_heat_observed_date=?, last_insemination_date=?, insemination_sire_tag_id=?, pregnancy_status=?, 
                                    pregnancy_diagnosis_date=?, expected_due_date=?, lactation_number=?, last_updated=CURRENT_TIMESTAMP
                                    WHERE cattle_id=? AND user_id=?
                                """, (tag_id_to_check or None, name_form_val.strip() or None, final_breed_to_save_val, sex_form_val, dob_form_val, purchase_date_form_val, purchase_price_form_val,
                                        current_status_form_val, notes_form_val, sire_id_form_val.strip() or None, dam_id_form_val.strip() or None,
                                        db_lcd_val.strftime('%Y-%m-%d') if db_lcd_val else None, db_lhd_val.strftime('%Y-%m-%d') if db_lhd_val else None,
                                        db_lid_val.strftime('%Y-%m-%d') if db_lid_val else None, db_insem_sire_val, db_preg_status_val,
                                        db_pdd_val.strftime('%Y-%m-%d') if db_pdd_val else None, db_edd_val_to_store.strftime('%Y-%m-%d') if db_edd_val_to_store else None,
                                        db_lact_num_val, editing_cattle_id_herd, st.session_state.user_id))
                                st.success(f"Animal '{name_form_val.strip() or tag_id_to_check}' updated successfully!")
                            else: # Adding new animal
                                cursor.execute("""
                                    INSERT INTO user_cattle (user_id, tag_id, name, breed, sex, dob, purchase_date, purchase_price, current_status, notes, sire_tag_id, dam_tag_id,
                                                           last_calving_date, last_heat_observed_date, last_insemination_date, insemination_sire_tag_id, pregnancy_status, 
                                                           pregnancy_diagnosis_date, expected_due_date, lactation_number)
                                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                                """, (st.session_state.user_id, tag_id_to_check or None, name_form_val.strip() or None, final_breed_to_save_val, sex_form_val, dob_form_val, purchase_date_form_val, purchase_price_form_val,
                                        current_status_form_val, notes_form_val, sire_id_form_val.strip() or None, dam_id_form_val.strip() or None,
                                        db_lcd_val.strftime('%Y-%m-%d') if db_lcd_val else None, db_lhd_val.strftime('%Y-%m-%d') if db_lhd_val else None,
                                        db_lid_val.strftime('%Y-%m-%d') if db_lid_val else None, db_insem_sire_val, db_preg_status_val,
                                        db_pdd_val.strftime('%Y-%m-%d') if db_pdd_val else None, db_edd_val_to_store.strftime('%Y-%m-%d') if db_edd_val_to_store else None,
                                        db_lact_num_val))
                                st.success(f"Animal '{name_form_val.strip() or tag_id_to_check}' added successfully!")
                            conn.commit()
                            st.rerun()
                        except sqlite3.IntegrityError: st.error(translate_text("DB Integrity Error: A unique constraint was violated (e.g. another animal has the same Tag ID).",current_lang))
                        except Exception as e_save: st.error(f"Error saving details: {e_save}")
    
    st.markdown("---")
    ts.subheader("📋 Your Cattle List")
    user_cattle_display_list_data = [] 
    try:
        cursor.execute("""
            SELECT cattle_id, tag_id, name, breed, sex, dob, current_status,
                   lactation_number, pregnancy_status, expected_due_date
            FROM user_cattle WHERE user_id = ? ORDER BY name, tag_id
        """, (st.session_state.user_id,))
        user_cattle_display_list_data = cursor.fetchall()

        if user_cattle_display_list_data:
            df_cattle_display_rows = []
            for row_data in user_cattle_display_list_data:
                (db_id_disp, tag_id_disp, name_disp, breed_disp, sex_disp, dob_disp, 
                 status_disp, lact_disp, preg_disp, edd_disp) = row_data
                dob_formatted_disp = datetime.strptime(dob_disp, '%Y-%m-%d').strftime('%d %b %Y') if dob_disp else "N/A"
                edd_formatted_disp = datetime.strptime(edd_disp, '%Y-%m-%d').strftime('%d %b %Y') if edd_disp else "N/A"
                age_str_disp = "N/A"
                if dob_disp:
                    try:
                        birth_date_disp = datetime.strptime(dob_disp, '%Y-%m-%d').date()
                        age_delta_disp = date.today() - birth_date_disp
                        years_disp, months_disp = age_delta_disp.days // 365, (age_delta_disp.days % 365) // 30
                        age_str_disp = f"{years_disp}y {months_disp}m"
                    except: pass
                df_cattle_display_rows.append([tag_id_disp, name_disp, breed_disp, sex_disp, age_str_disp, status_disp, lact_disp, preg_disp, edd_formatted_disp])
            df_to_display_on_page = pd.DataFrame(df_cattle_display_rows, columns=["Tag ID", "Name", "Breed", "Sex", "Age", "Status", "Lactation#", "Pregnancy", "EDD"])
            st.dataframe(df_to_display_on_page, use_container_width=True, hide_index=True)

            if user_cattle_display_list_data:
                excel_herd_bytes = generate_herd_details_excel(user_cattle_display_list_data, st.session_state.username)
                st.download_button(
                    label="📄 Download Herd Details (Excel)",
                    data=excel_herd_bytes,
                    file_name=f"kamdhenu_herd_{st.session_state.username}_{date.today().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_full_herd_excel_btn_myherd"
                )
        else:
            st.info(translate_text("No cattle added yet. Use the 'Add / Edit Animal Details' section.",current_lang))
    except sqlite3.Error as e_fetch_list_display:
        st.error(f"Could not fetch herd list for display: {e_fetch_list_display}")

    if user_cattle_display_list_data:
        st.markdown("---")
        ts.subheader("🗂️ Manage Individual Animal Records")
        detail_animal_options_map_herd = {
            f"{row[2] or 'Unnamed'} (Tag: {row[1] or 'N/A'}, ID: {row[0]})": row[0] 
            for row in user_cattle_display_list_data
        }
        
        # Pre-selection logic for the selectbox for tabs
        default_animal_selectbox_index_tabs = 0 
        if pre_selected_animal_id_from_reminders:
            options_list_for_tabs = ["--- Select Animal ---"] + list(detail_animal_options_map_herd.keys())
            for i, display_name_option in enumerate(options_list_for_tabs):
                if display_name_option != "--- Select Animal ---":
                    if detail_animal_options_map_herd.get(display_name_option) == pre_selected_animal_id_from_reminders:
                        default_animal_selectbox_index_tabs = i
                        break
        
        selected_animal_display_name_herd = st.selectbox(translate_text("Select Animal for Detailed Records:",current_lang),
            options=["--- Select Animal ---"] + list(detail_animal_options_map_herd.keys()),
            index=default_animal_selectbox_index_tabs, 
            key="herd_detail_animal_select_v5"
        )

        if selected_animal_display_name_herd != "--- Select Animal ---":
            selected_cattle_id_detail_herd = detail_animal_options_map_herd[selected_animal_display_name_herd]
            active_tab_key = f"my_herd_active_tab_for_animal_{selected_cattle_id_detail_herd}" # Key for session state

            try:
                cursor.execute("""SELECT tag_id, name, breed, sex, dob, purchase_date, purchase_price,
                                        current_status, notes, sire_tag_id, dam_tag_id,
                                        last_calving_date, last_heat_observed_date, last_insemination_date,
                                        insemination_sire_tag_id, pregnancy_status, pregnancy_diagnosis_date,
                                        expected_due_date, lactation_number
                                FROM user_cattle WHERE cattle_id = ? AND user_id = ?""",
                               (selected_cattle_id_detail_herd, st.session_state.user_id))
                animal_data_for_tabs = cursor.fetchone()
            except sqlite3.Error as e_detail_fetch:
                st.error(f"Error fetching details for selected animal: {e_detail_fetch}")
                animal_data_for_tabs = None

            if animal_data_for_tabs:
                (tag_id_tab, name_tab, breed_tab, sex_tab, dob_tab, _, _, 
                 status_tab, notes_tab, sire_tab, dam_tab,
                 lcd_tab, lhod_tab, lid_tab, insem_sire_tab,
                 preg_status_tab, pdd_tab, edd_tab, lact_num_tab) = animal_data_for_tabs

                animal_info_for_report = {
                    "cattle_id": selected_cattle_id_detail_herd, "user_id": st.session_state.user_id,
                    "tag_id": tag_id_tab, "name": name_tab, "breed": breed_tab,
                    "sex": sex_tab, "dob": dob_tab
                }
                is_calf_for_tabs_val = False
                if dob_tab:
                    try:
                        birth_date_tab = datetime.strptime(dob_tab, "%Y-%m-%d").date()
                        if (date.today() - birth_date_tab).days < 270: is_calf_for_tabs_val = True
                    except: pass
                if not is_calf_for_tabs_val and sex_tab and "calf" in sex_tab.lower(): is_calf_for_tabs_val = True

                tab_titles_list_val = ["📝 Basic & Breeding", "💉 Vaccinations Log", "❤️‍🩹 Health Events Log", "🔔 Animal Reminders", "🥛 Milk Log"]
                if is_calf_for_tabs_val: tab_titles_list_val.append("👶 Calf Rearing")
                
                current_default_tab_index = 0 
                if pre_selected_tab_name_from_reminders and pre_selected_tab_name_from_reminders in tab_titles_list_val:
                    current_default_tab_index = tab_titles_list_val.index(pre_selected_tab_name_from_reminders)
                
                # Store and retrieve active tab index from session state
                if active_tab_key not in st.session_state:
                     st.session_state[active_tab_key] = current_default_tab_index

                # This part is tricky with st.tabs as it doesn't directly support setting default_index after creation.
                # The pre-selection of animal is the primary way to navigate. For tabs, user might need to click.
                # However, we use active_tab_key to remember selection if actions within tabs cause reruns.
                tabs_obj_herd = st.tabs(tab_titles_list_val)

                # Manually switch to the intended tab if navigated (this is a conceptual workaround)
                # For actual st.tabs, this won't change the visual active tab directly after initial render.
                # The effect is more for when the page re-renders due to an action within the tabs.
                # selected_tab_visual_index = st.session_state[active_tab_key]


                with tabs_obj_herd[0]: # Basic & Breeding Info
                    # ... (Your existing Basic & Breeding info display) ...
                    ts.subheader(f"Details for: {name_tab or tag_id_tab}") # Make sure these variables are defined
                    ts.markdown(f"**Breed:** {breed_tab or 'N/A'} | **Sex:** {sex_tab or 'N/A'} | **DOB:** {dob_tab or 'N/A'}")
                    ts.markdown(f"**Status:** {status_tab or 'N/A'}")
                    if notes_tab: st.markdown(f"**Notes:** {notes_tab}")
                    if sire_tab or dam_tab: st.markdown(f"**Pedigree:** Sire - {sire_tab or 'N/A'}, Dam - {dam_tab or 'N/A'}")
                    if sex_tab in ['Female', 'Cow', 'Heifer']:
                        ts.markdown("###### Breeding Cycle Summary:")
                        if lact_num_tab is not None: ts.write(f"**Lactation Number:** {lact_num_tab}")
                        if lcd_tab: ts.write(f"**Last Calved:** {datetime.strptime(lcd_tab, '%Y-%m-%d').strftime('%d %b %Y')}")
                        if lhod_tab: ts.write(f"**Last Heat:** {datetime.strptime(lhod_tab, '%Y-%m-%d').strftime('%d %b %Y')}")
                        if lid_tab: ts.write(f"**Last Inseminated:** {datetime.strptime(lid_tab, '%Y-%m-%d').strftime('%d %b %Y')} (Sire: {insem_sire_tab or 'N/A'})")
                        if preg_status_tab: ts.write(f"**Pregnancy Status:** {preg_status_tab}")
                        if pdd_tab: ts.write(f"**Preg. Diagnosis Date:** {datetime.strptime(pdd_tab, '%Y-%m-%d').strftime('%d %b %Y')}")
                        if edd_tab: st.success(f"**Expected Due Date:** {datetime.strptime(edd_tab, '%Y-%m-%d').strftime('%d %b %Y')}")
                    else: st.caption(translate_text("Breeding info not applicable.",current_lang))


                with tabs_obj_herd[1]: # Vaccinations Log
                    ts.subheader(f"💉 Vaccination Records for {name_tab or tag_id_tab}")
                    with st.expander(translate_text("➕ Add New Vaccination Record",current_lang), expanded=False):
                        with st.form(translate_text(f"vaccination_form_{selected_cattle_id_detail_herd}_v3",current_lang), clear_on_submit=True): # Incremented key
                            vacc_name = st.text_input(translate_text("Vaccination Name*",current_lang), key=f"vacc_name_{selected_cattle_id_detail_herd}_v3")
                            vacc_date_val = st.date_input(translate_text("Date Administered*",current_lang), date.today(), key=f"vacc_date_val_{selected_cattle_id_detail_herd}_v3") # Renamed variable
                            vacc_next_due = st.date_input(translate_text("Next Due Date (Optional)",current_lang), value=None, key=f"vacc_next_due_{selected_cattle_id_detail_herd}_v3")
                            vacc_batch = st.text_input(translate_text("Batch Number",current_lang), key=f"vacc_batch_{selected_cattle_id_detail_herd}_v3")
                            vacc_admin_by = st.text_input(translate_text("Administered By",current_lang), key=f"vacc_admin_by_{selected_cattle_id_detail_herd}_v3")
                            vacc_notes = st.text_area(translate_text("Notes",current_lang), key=f"vacc_notes_{selected_cattle_id_detail_herd}_v3")
                            
                            submit_vacc = st.form_submit_button(translate_text("Save Vaccination Record",current_lang))
                            if submit_vacc:
                                if not vacc_name: st.error(translate_text("Vaccination Name is required.",current_lang))
                                else:
                                    try:
                                        cursor.execute("""
                                            INSERT INTO vaccinations_log (cattle_id, user_id, vaccination_name, vaccination_date, next_due_date, batch_number, administered_by, notes)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                        """, (selected_cattle_id_detail_herd, st.session_state.user_id, vacc_name, vacc_date_val.strftime('%Y-%m-%d'), # Used vacc_date_val
                                              vacc_next_due.strftime('%Y-%m-%d') if vacc_next_due else None,
                                              vacc_batch or None, vacc_admin_by or None, vacc_notes or None))
                                        last_row_id = cursor.lastrowid
                                        if vacc_next_due:
                                            reminder_desc_vacc = f"Vaccination '{vacc_name}' due"
                                            try:
                                                cursor.execute("""
                                                    INSERT INTO health_reminders_status (user_id, cattle_id, original_log_id, reminder_type, reminder_description, due_date, status)
                                                    VALUES (?, ?, ?, ?, ?, ?, 'pending')
                                                """, (st.session_state.user_id, selected_cattle_id_detail_herd, last_row_id, 
                                                      'vaccination_due', reminder_desc_vacc, vacc_next_due.strftime('%Y-%m-%d')))
                                                logger.info(f"Reminder created for vaccination {vacc_name} for cattle {selected_cattle_id_detail_herd}")
                                            except sqlite3.Error as e_rem_create:
                                                logger.error(f"Failed to create reminder for vaccination: {e_rem_create}")
                                                st.warning(translate_text("Vaccination saved, but failed to create reminder.",current_lang), icon="⚠️")
                                        conn.commit()
                                        st.success(translate_text("Vaccination record saved! Reminder created if due date was set.",current_lang));
                                        st.session_state[active_tab_key] = 1
                                        st.rerun()
                                    except sqlite3.Error as e_vacc_save: st.error(f"DB error saving vaccination: {e_vacc_save}")
                            
                    ts.markdown("##### 📖 Existing Vaccination Records:")
                    try:
                        cursor.execute("""
                            SELECT log_id, vaccination_name, vaccination_date, next_due_date, batch_number, administered_by, notes
                            FROM vaccinations_log WHERE cattle_id = ? AND user_id = ? ORDER BY vaccination_date DESC
                        """, (selected_cattle_id_detail_herd, st.session_state.user_id))
                        vacc_records_full = cursor.fetchall()
                        vacc_records_display = []
                        for r_vacc_disp in vacc_records_full: # Renamed loop var
                            # log_id (r[0]) is kept if needed for edit/delete, but not displayed in table directly
                            # For display, format dates:
                            date_admin_str = datetime.strptime(r_vacc_disp[2], '%Y-%m-%d').strftime('%d %b %Y') if r_vacc_disp[2] else "N/A"
                            next_due_str = datetime.strptime(r_vacc_disp[3], '%Y-%m-%d').strftime('%d %b %Y') if r_vacc_disp[3] else "N/A"
                            vacc_records_display.append((r_vacc_disp[1], date_admin_str, next_due_str, r_vacc_disp[4], r_vacc_disp[5], r_vacc_disp[6]))


                        if vacc_records_display:
                            df_vacc = pd.DataFrame(vacc_records_display, columns=["Vaccine", "Date", "Next Due", "Batch#", "Admin By", "Notes"])
                            st.dataframe(df_vacc, hide_index=True, use_container_width=True)

                            report_cols_vacc = st.columns(2)
                            with report_cols_vacc[0]:
                                pdf_buffer = generate_animal_vaccination_report_pdf(animal_info_for_report, vacc_records_display)
                                st.download_button(
                                    label="📄 Download Report (PDF)", data=pdf_buffer,
                                    file_name=f"vacc_report_{animal_info_for_report.get('tag_id','animal')}_{date.today().strftime('%Y%m%d')}.pdf",
                                    mime="application/pdf", key=f"download_vacc_report_pdf_{selected_cattle_id_detail_herd}_v3", use_container_width=True
                                )
                            with report_cols_vacc[1]:
                                excel_vacc_bytes = generate_animal_vaccination_report_excel(animal_info_for_report, vacc_records_display)
                                st.download_button(
                                    label="📊 Download Report (Excel)", data=excel_vacc_bytes,
                                    file_name=f"vacc_report_{animal_info_for_report.get('tag_id','animal')}_{date.today().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"download_vacc_report_excel_{selected_cattle_id_detail_herd}_v3", use_container_width=True
                                )
                        else: st.info(translate_text("No vaccination records for this animal.",current_lang))
                    except sqlite3.Error as e_fetch_vacc: st.error(f"Error fetching vaccination records: {e_fetch_vacc}")

                with tabs_obj_herd[2]: # Health Events Log
                    # ... (Your Health Events Log code, including the auto-reminder creation like in Vaccinations log) ...
                    ts.subheader(f"❤️‍🩹 Health Event Records for {name_tab or tag_id_tab}")
                    with st.expander(translate_text("➕ Add New Health Event Record",current_lang), expanded=False):
                        with st.form(translate_text(f"health_event_form_{selected_cattle_id_detail_herd}_v3",current_lang), clear_on_submit=True): # Incremented key
                            he_type_options = ['Illness', 'Treatment', 'Routine Checkup', 'Deworming', 'Injury', 'Other']
                            he_event_type = st.selectbox(translate_text("Event Type*",current_lang), options=he_type_options, key=f"he_type_{selected_cattle_id_detail_herd}_v3")
                            he_event_date_val = st.date_input(translate_text("Event Date*",current_lang), date.today(), key=f"he_date_val_{selected_cattle_id_detail_herd}_v3") # Renamed
                            he_symptoms = st.text_area(translate_text("Symptoms Observed",current_lang), key=f"he_symptoms_{selected_cattle_id_detail_herd}_v3")
                            he_diagnosis = st.text_input(translate_text("Diagnosis",current_lang), key=f"he_diag_{selected_cattle_id_detail_herd}_v3")
                            he_treatment = st.text_area(translate_text("Treatment Administered",current_lang), key=f"he_treat_{selected_cattle_id_detail_herd}_v3")
                            he_vet = st.text_input(translate_text("Veterinarian Involved",current_lang), key=f"he_vet_{selected_cattle_id_detail_herd}_v3")
                            he_next_checkup = st.date_input(translate_text("Next Checkup Date (Optional)",current_lang), value=None, key=f"he_next_checkup_{selected_cattle_id_detail_herd}_v3")
                            he_outcome = st.text_input(translate_text("Outcome",current_lang), key=f"he_outcome_{selected_cattle_id_detail_herd}_v3")
                            he_notes = st.text_area(translate_text("Additional Notes",current_lang), key=f"he_notes_{selected_cattle_id_detail_herd}_v3")
                            submit_he = st.form_submit_button(translate_text("Save Health Event Record",current_lang))
                            if submit_he:
                                try:
                                    cursor.execute("""
                                        INSERT INTO health_events_log (cattle_id, user_id, event_type, event_date, symptoms_observed, diagnosis, treatment_administered, veterinarian_involved, next_checkup_date, outcome, notes)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """, (selected_cattle_id_detail_herd, st.session_state.user_id, he_event_type, he_event_date_val.strftime('%Y-%m-%d'), # Used he_event_date_val
                                          he_symptoms or None, he_diagnosis or None, he_treatment or None, he_vet or None,
                                          he_next_checkup.strftime('%Y-%m-%d') if he_next_checkup else None,
                                          he_outcome or None, he_notes or None))
                                    last_event_id = cursor.lastrowid
                                    if he_next_checkup:
                                        reminder_desc_he = f"{he_event_type} follow-up/checkup due"
                                        try:
                                            cursor.execute("""
                                                INSERT INTO health_reminders_status (user_id, cattle_id, original_log_id, reminder_type, reminder_description, due_date, status)
                                                VALUES (?, ?, ?, ?, ?, ?, 'pending')
                                            """, (st.session_state.user_id, selected_cattle_id_detail_herd, last_event_id, 'health_checkup_due', reminder_desc_he, he_next_checkup.strftime('%Y-%m-%d')))
                                            logger.info(f"Reminder created for health event {he_event_type} for cattle {selected_cattle_id_detail_herd}")
                                        except sqlite3.Error as e_rem_create_he:
                                            logger.error(f"Failed to create reminder for health event: {e_rem_create_he}")
                                            st.warning(translate_text("Health event saved, but failed to create reminder.",current_lang), icon="⚠️")
                                    conn.commit()
                                    st.success(translate_text("Health event record saved! Reminder created if checkup date was set.",current_lang));
                                    st.session_state[active_tab_key] = 2
                                    st.rerun()
                                except sqlite3.Error as e_he_save: st.error(f"DB error saving health event: {e_he_save}")
                    ts.markdown("##### 📖 Existing Health Event Records:")
                    try:
                        cursor.execute("""
                            SELECT event_date, event_type, symptoms_observed, diagnosis, treatment_administered, veterinarian_involved, next_checkup_date, outcome, notes
                            FROM health_events_log WHERE cattle_id = ? AND user_id = ? ORDER BY event_date DESC
                        """, (selected_cattle_id_detail_herd, st.session_state.user_id))
                        he_records = cursor.fetchall()
                        if he_records:
                            df_he = pd.DataFrame(he_records, columns=["Date", "Type", "Symptoms", "Diagnosis", "Treatment", "Vet", "Next Checkup", "Outcome", "Notes"])
                            df_he['Date'] = pd.to_datetime(df_he['Date']).dt.strftime('%d %b %Y')
                            df_he['Next Checkup'] = pd.to_datetime(df_he['Next Checkup'], errors='coerce').dt.strftime('%d %b %Y')
                            st.dataframe(df_he[["Date", "Type", "Diagnosis", "Treatment", "Vet", "Outcome", "Next Checkup"]], hide_index=True, use_container_width=True)
                        else: st.info(translate_text("No health event records for this animal.",current_lang))
                    except sqlite3.Error as e_fetch_he: st.error(f"Error fetching health event records: {e_fetch_he}")
                
                with tabs_obj_herd[3]: # Animal Reminders Tab
                    ts.subheader(f"🔔 Health Reminders for {name_tab or tag_id_tab}")
                    def handle_specific_animal_reminder_action(reminder_id_param, new_status_param, notes_param=""):
                        try:
                            action_taken_on_local_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            cursor.execute("""
                                UPDATE health_reminders_status SET status = ?, action_taken_on = ?, action_notes = ?
                                WHERE reminder_id = ? AND user_id = ? AND cattle_id = ?
                            """, (new_status_param, action_taken_on_local_str, notes_param, 
                                  reminder_id_param, st.session_state.user_id, selected_cattle_id_detail_herd))
                            conn.commit()
                            st.toast(translate_text(f"Reminder for {name_tab or tag_id_tab} marked as {new_status_param}.",current_lang), icon="✅")
                            st.session_state[active_tab_key] = 3 
                            st.rerun()
                        except sqlite3.Error as e_action_specific: st.error(f"Error updating reminder: {e_action_specific}")
                    ts.markdown("##### Pending Reminders for this Animal:")
                    try:
                        cursor.execute("""
                            SELECT reminder_id, reminder_description, due_date 
                            FROM health_reminders_status WHERE cattle_id = ? AND user_id = ? AND status = 'pending' ORDER BY due_date ASC
                        """, (selected_cattle_id_detail_herd, st.session_state.user_id))
                        pending_animal_reminders = cursor.fetchall()
                        if pending_animal_reminders:
                            for r_id_animal, r_desc_animal, r_due_animal in pending_animal_reminders:
                                due_date_obj_animal = datetime.strptime(r_due_animal, '%Y-%m-%d').date()
                                due_date_fmt_animal = due_date_obj_animal.strftime('%d %b %Y')
                                is_overdue_animal = due_date_obj_animal < date.today()
                                is_due_today_animal = due_date_obj_animal == date.today()
                                reminder_text_html = ""
                                if is_overdue_animal: reminder_text_html = f"<span style='color:red;'>🚨 **OVERDUE:** {r_desc_animal} - Was due: **{due_date_fmt_animal}**</span>"
                                elif is_due_today_animal: reminder_text_html = f"<span style='color:orange;'>🗓️ **DUE TODAY:** {r_desc_animal} - Due: **{due_date_fmt_animal}**</span>"
                                else: reminder_text_html = f"<span style='color:blue;'>🔔 **UPCOMING:** {r_desc_animal} - Due: **{due_date_fmt_animal}**</span>"
                                st.markdown(reminder_text_html, unsafe_allow_html=True)
                                act_cols_animal_tab = st.columns(2)
                                with act_cols_animal_tab[0]:
                                    if st.button(translate_text("✅ Mark Completed",current_lang, key="auto_btn_6"), key=f"complete_anim_tab_rem_{r_id_animal}", type="primary", use_container_width=True):
                                        handle_specific_animal_reminder_action(r_id_animal, "completed", f"Completed for {name_tab or tag_id_tab}")
                                with act_cols_animal_tab[1]:
                                    if st.button(translate_text("🚫 Dismiss",current_lang, key="auto_btn_7"), key=f"dismiss_anim_tab_rem_{r_id_animal}", use_container_width=True):
                                        handle_specific_animal_reminder_action(r_id_animal, "dismissed", f"Dismissed for {name_tab or tag_id_tab}")
                                st.markdown("---")
                        else: st.info(translate_text("No pending reminders for this animal.",current_lang))
                    except sqlite3.Error as e_fetch_anim_rem: st.error(f"Error fetching animal-specific reminders: {e_fetch_anim_rem}")
                    ts.markdown("##### Handled Reminders for this Animal (Last 30 days):")
                    try:
                        cursor.execute("""
                            SELECT reminder_description, due_date, status, action_taken_on, action_notes
                            FROM health_reminders_status WHERE cattle_id = ? AND user_id = ? AND status IN ('completed', 'dismissed')
                                  AND date(action_taken_on) >= date('now', '-30 days') ORDER BY action_taken_on DESC
                        """, (selected_cattle_id_detail_herd, st.session_state.user_id))
                        handled_animal_reminders = cursor.fetchall()
                        if handled_animal_reminders:
                            df_handled_animal = pd.DataFrame(handled_animal_reminders, columns=["Reminder", "Original Due", "Status", "Handled On", "Notes"])
                            df_handled_animal["Original Due"] = pd.to_datetime(df_handled_animal["Original Due"]).dt.strftime('%d %b %Y')
                            df_handled_animal["Handled On"] = pd.to_datetime(df_handled_animal["Handled On"]).dt.strftime('%d %b %Y, %I:%M %p')
                            st.dataframe(df_handled_animal, hide_index=True, use_container_width=True)
                        else: st.info(translate_text("No reminders handled for this animal in the last 30 days.",current_lang))
                    except sqlite3.Error as e_fetch_handled_anim: st.error(f"Error fetching handled animal reminders: {e_fetch_handled_anim}")

                milk_log_tab_index = 4 
                calf_tab_index_val = 5
                if sex_tab in ['Cow', 'Female', 'Heifer']:
                    if milk_log_tab_index < len(tabs_obj_herd):
                        with tabs_obj_herd[milk_log_tab_index]:
                            ts.subheader(f"Milk Production Log for {name_tab or tag_id_tab}")
                            ts.subheader(f"🥛 Milk Production Log for {name_tab or tag_id_tab}")

                            # --- Form to Add New Milk Log Entry ---
                            with st.expander(translate_text("➕ Add New Milk Record",current_lang), expanded=False):
                                # Ensure selected_cattle_id_detail_herd and st.session_state.user_id are available from the outer scope
                                with st.form(translate_text(f"milk_log_form_{selected_cattle_id_detail_herd}_v2",current_lang), clear_on_submit=True): # Incremented key
                                    ml_date_val = st.date_input(translate_text("Log Date*",current_lang), date.today(), key=f"ml_date_{selected_cattle_id_detail_herd}_v2")
                                    ml_session_options = ["Morning", "Afternoon", "Evening", "Full Day", "Other"]
                                    ml_session_val = st.selectbox(translate_text("Milking Session*",current_lang), ml_session_options, key=f"ml_session_{selected_cattle_id_detail_herd}_v2")
                                    ml_yield_val = st.number_input(translate_text("Milk Yield (Liters)*",current_lang), min_value=0.01, value=5.0, step=0.1, help="Yield must be greater than 0.", key=f"ml_yield_{selected_cattle_id_detail_herd}_v2")
                                    
                                    ml_col1, ml_col2 = st.columns(2)
                                    with ml_col1:
                                        ml_fat_val = st.number_input(translate_text("Fat % (Optional)",current_lang), min_value=0.0, max_value=15.0, step=0.01, value=None, placeholder="e.g., 3.50", key=f"ml_fat_{selected_cattle_id_detail_herd}_v2")
                                    with ml_col2:
                                        ml_snf_val = st.number_input(translate_text("SNF % (Optional)",current_lang), min_value=0.0, max_value=15.0, step=0.01, value=None, placeholder="e.g., 8.50", key=f"ml_snf_{selected_cattle_id_detail_herd}_v2")
                                    
                                    ml_notes_val = st.text_area(translate_text("Notes (Optional)",current_lang), key=f"ml_notes_{selected_cattle_id_detail_herd}_v2")
                                    
                                    submit_milk_log_btn = st.form_submit_button(translate_text("Save Milk Record",current_lang))

                                    if submit_milk_log_btn:
                                        if not (ml_yield_val and ml_yield_val > 0): # Basic validation
                                            st.error(translate_text("Milk Yield must be entered and be greater than 0.",current_lang))
                                        else:
                                            try:
                                                # Ensure cursor and conn are available from the outer "My Herd" scope
                                                cursor.execute("""
                                                    INSERT INTO milk_log (cattle_id, user_id, log_date, milking_session, milk_yield_liters, fat_percentage, snf_percentage, notes)
                                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                                """, (selected_cattle_id_detail_herd, st.session_state.user_id, ml_date_val.strftime('%Y-%m-%d'),
                                                      ml_session_val, ml_yield_val, ml_fat_val, ml_snf_val, ml_notes_val.strip() or None))
                                                conn.commit()
                                                st.success(translate_text("Milk record saved successfully!",current_lang))
                                                # To stay on the milk log tab after submission:
                                                if active_tab_key in st.session_state: # Ensure key exists
                                                    st.session_state[active_tab_key] = milk_log_tab_index
                                                st.rerun() 
                                            except sqlite3.IntegrityError as ie:
                                                st.error(f"Database Integrity Error: {ie}. This could mean the 'milk_log' table doesn't exist or there's a schema mismatch. Please contact support or ensure DB setup is correct.")
                                                logger.error(f"Milk log save IntegrityError for cattle {selected_cattle_id_detail_herd} by user {st.session_state.user_id}: {ie}")
                                            except sqlite3.OperationalError as oe:
                                                 st.error(f"Database Operational Error: {oe}. The 'milk_log' table might be missing. Please ensure it is created.")
                                                 logger.error(f"Milk log save OperationalError for cattle {selected_cattle_id_detail_herd} by user {st.session_state.user_id}: {oe}")
                                            except sqlite3.Error as e_ml_save:
                                                st.error(f"Database error saving milk record: {e_ml_save}")
                                                logger.error(f"Milk log save DB error for cattle {selected_cattle_id_detail_herd} by user {st.session_state.user_id}: {e_ml_save}")
                            
                            st.markdown("---")
                            ts.markdown("##### 📖 Existing Milk Records (Last 30 entries):")
                            try:
                                # Ensure cursor is available
                                cursor.execute("""
                                    SELECT log_id, log_date, milking_session, milk_yield_liters, fat_percentage, snf_percentage, notes
                                    FROM milk_log 
                                    WHERE cattle_id = ? AND user_id = ? 
                                    ORDER BY log_date DESC, recorded_at DESC 
                                    LIMIT 30 
                                """, (selected_cattle_id_detail_herd, st.session_state.user_id))
                                milk_records_data = cursor.fetchall()

                                if milk_records_data:
                                    # Include log_id for potential future edit/delete functionality, but don't display it by default
                                    df_milk_display = pd.DataFrame(milk_records_data, columns=["Log ID", "Date", "Session", "Yield (L)", "Fat %", "SNF %", "Notes"])
                                    
                                    # Format date for display
                                    df_milk_display['Date'] = pd.to_datetime(df_milk_display['Date']).dt.strftime('%d %b %Y')
                                    
                                    # Format numbers and fill NaN for display
                                    for col_format in ["Yield (L)", "Fat %", "SNF %"]:
                                        df_milk_display[col_format] = pd.to_numeric(df_milk_display[col_format], errors='coerce').round(2)
                                    df_milk_display = df_milk_display.fillna('N/A')
                                    
                                    st.dataframe(df_milk_display[["Date", "Session", "Yield (L)", "Fat %", "SNF %", "Notes"]], hide_index=True, use_container_width=True)

                                    # TODO: Add Delete functionality for milk records if needed
                                    # with st.expander("Delete a Milk Record (by Log ID)"):
                                    #     delete_log_id = st.number_input("Enter Log ID to Delete", min_value=1, step=1, value=None)
                                    #     if st.button("Delete Record", key=f"delete_milk_log_{selected_cattle_id_detail_herd}"):
                                    #         if delete_log_id:
                                    #             # Add deletion logic here
                                    #             st.warning("Deletion logic not yet implemented in this snippet.")
                                    #         else:
                                    #             st.error("Please enter a Log ID.")
                                else:
                                    st.info(translate_text("No milk records found for this animal yet.",current_lang))  
                            
                            except sqlite3.OperationalError:
                                st.warning(translate_text("The 'milk_log' table might not exist or is inaccessible. Please ensure the database is set up correctly for this feature.",current_lang))
                                logger.warning(f"Milk log display: milk_log table not found or schema issue for user {st.session_state.user_id}, cattle {selected_cattle_id_detail_herd}.")
                            except sqlite3.Error as e_fetch_ml:
                                st.error(translate_text(f"Error fetching milk records: {e_fetch_ml}",current_lang))
                                logger.error(f"Error fetching milk_records for user {st.session_state.user_id}, cattle {selected_cattle_id_detail_herd}: {e_fetch_ml}")
                else: 
                    if milk_log_tab_index < len(tabs_obj_herd):
                        with tabs_obj_herd[milk_log_tab_index]:
                            st.info(translate_text("Milk log not applicable for this animal type/sex.",current_lang))
                
                if is_calf_for_tabs_val:
                    if calf_tab_index_val < len(tabs_obj_herd):
                        with tabs_obj_herd[calf_tab_index_val]:
                            ts.subheader(f"Calf Rearing Log for {name_tab or tag_id_tab}")
                            if not calf_rearing_table_exists(cursor): 
                                st.warning(translate_text("Calf rearing log table missing.",current_lang))
                            else:
                                with st.form(translate_text(f"calf_log_form_{selected_cattle_id_detail_herd}_v5",current_lang), clear_on_submit=True): #Key v5
                                    ts.markdown("##### Add New Calf Log Entry")
                                    log_date_calf_form = st.date_input(translate_text("Log Date",current_lang), date.today(), key=f"calflog_date_{selected_cattle_id_detail_herd}_v5")
                                    clf_c1, clf_c2 = st.columns(2)
                                    colostrum_fed_form = clf_c1.checkbox(translate_text("Colostrum Fed (within 6h)?",current_lang), key=f"calflog_colos_{selected_cattle_id_detail_herd}_v5")
                                    colostrum_amount_form = clf_c2.number_input(translate_text("Colostrum Amount (L)",current_lang), min_value=0.0, value=2.0, step=0.1, key=f"calflog_colosamt_{selected_cattle_id_detail_herd}_v5")
                                    milk_rep_type_form = st.text_input(translate_text("Milk/Replacer Type",current_lang), key=f"calflog_milktype_{selected_cattle_id_detail_herd}_v5")
                                    milk_rep_amount_form = st.number_input(translate_text("Milk/Replacer Amt (L/day)",current_lang), min_value=0.0, value=4.0, step=0.1, key=f"calflog_milkamt_{selected_cattle_id_detail_herd}_v5")
                                    starter_intake_form = st.number_input(translate_text("Starter Intake (g/day)",current_lang), min_value=0.0, value=100.0, step=10.0, key=f"calflog_starter_{selected_cattle_id_detail_herd}_v5")
                                    weight_calf_form = st.number_input(translate_text("Weight (kg, Optional)",current_lang), min_value=0.0, step=0.5, key=f"calflog_weight_{selected_cattle_id_detail_herd}_v5")
                                    deworm_done_form = st.checkbox(translate_text("Deworming Done (on log date)?",current_lang), key=f"calflog_deworm_{selected_cattle_id_detail_herd}_v5")
                                    deworm_prod_form = st.text_input(translate_text("Deworming Product",current_lang), key=f"calflog_dewormprod_{selected_cattle_id_detail_herd}_v5")
                                    vacc_given_calf_form = st.text_input(translate_text("Vaccination Given (Note)",current_lang), key=f"calflog_vacc_{selected_cattle_id_detail_herd}_v5")
                                    health_notes_calf_form = st.text_area(translate_text("Health Notes/Observations",current_lang), key=f"calflog_hnotes_{selected_cattle_id_detail_herd}_v5")
                                    if st.form_submit_button(translate_text("Save Calf Log",current_lang)):
                                        try:
                                            cursor.execute("""INSERT INTO calf_rearing_log
                                                              (cattle_id, log_date, colostrum_fed_within_6h, colostrum_amount_liters,
                                                               milk_replacer_type, milk_replacer_amount_liters, starter_feed_intake_grams,
                                                               deworming_done, deworming_product, vaccination_given, weight_kg, health_notes)
                                                              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                                           (selected_cattle_id_detail_herd, log_date_calf_form.strftime('%Y-%m-%d'),
                                                            1 if colostrum_fed_form else 0, colostrum_amount_form if colostrum_fed_form and colostrum_amount_form > 0 else None,
                                                            milk_rep_type_form or None, milk_rep_amount_form if milk_rep_amount_form > 0 else None, starter_intake_form if starter_intake_form > 0 else None,
                                                            1 if deworm_done_form else 0, deworm_prod_form.strip() or None if deworm_done_form else None,
                                                            vacc_given_calf_form.strip() or None, weight_calf_form if weight_calf_form > 0 else None, health_notes_calf_form.strip() or None))
                                            conn.commit()
                                            st.success(translate_text("Calf log entry saved!",current_lang))
                                            st.session_state[active_tab_key] = calf_tab_index_val # Stay on calf tab
                                            st.rerun()
                                        except sqlite3.Error as e_crlog_save: st.error(f"DB error saving calf log: {e_crlog_save}")
                                ts.markdown("##### 📖 Existing Calf Log Entries")
                                try:
                                    cursor.execute("""SELECT log_date, colostrum_fed_within_6h, colostrum_amount_liters,
                                                      milk_replacer_type, milk_replacer_amount_liters, starter_feed_intake_grams,
                                                      weight_kg, deworming_done, deworming_product, vaccination_given, health_notes
                                                      FROM calf_rearing_log WHERE cattle_id = ? ORDER BY log_date DESC""", (selected_cattle_id_detail_herd,))
                                    calf_logs_display_data = cursor.fetchall()
                                    if calf_logs_display_data:
                                        df_clogs_disp = pd.DataFrame(calf_logs_display_data, columns=["Date", "Colostrum Fed?", "Colostrum (L)", "Milk/Rep Type", "Milk/Rep Amt (L)", "Starter (g)", "Weight (kg)", "Dewormed?", "Deworm Prod.", "Vaccine Note", "Health Notes"])
                                        df_clogs_disp["Colostrum Fed?"] = df_clogs_disp["Colostrum Fed?"].apply(lambda x: "Yes" if x==1 else "No")
                                        df_clogs_disp["Dewormed?"] = df_clogs_disp["Dewormed?"].apply(lambda x: "Yes" if x==1 else "No")
                                        st.dataframe(df_clogs_disp, hide_index=True, use_container_width=True)
                                    else: st.info(translate_text("No rearing log entries for this calf.",current_lang))
                                except sqlite3.Error as e_fetch_clogs_disp: st.error(f"Error fetching calf logs: {e_fetch_clogs_disp}")
            else:
                st.info(translate_text("Select an animal from the list above to view or manage its detailed records.",current_lang))
    else:
        st.info(translate_text("No cattle in your herd yet. Add animals using the section above.",current_lang))

elif st.session_state.current_page == "My Profile":
    if not st.session_state.logged_in or st.session_state.role != 'farmer':
        st.warning(translate_text("Access denied. Please log in as a farmer.", current_lang))
        # ... (optional login button) ...
    else:
        st.title(translate_text(f"👤 My Profile - {st.session_state.username}", current_lang))
        st.markdown(translate_text("Update your farm details and product offerings.", current_lang))
        st.markdown("---")

        conn = get_connection()
        if not conn: 
            st.error(translate_text("Database connection failed.", current_lang))
            st.stop()
        cursor = conn.cursor()

        try:
            # Add 'upi_id' to the SELECT query
            cursor.execute("""SELECT full_name, email, phone_number, address, region,
                                    latitude, longitude, sells_products, product_categories, share_contact_info, upi_id
                              FROM users WHERE user_id = ?""", (st.session_state.user_id,))
            profile_data = cursor.fetchone()
        except sqlite3.Error as e:
            st.error(translate_text(f"Error fetching profile: {e}", current_lang))
            profile_data = None

        if profile_data:
            (p_fname, p_email, p_phone, p_address, p_region,
             p_lat, p_lon, p_sells, p_prod_cat, p_share_contact, p_upi_id) = profile_data # Unpack p_upi_id here

            with st.form("my_profile_form"):
                st.subheader(translate_text("Contact & Location Information", current_lang))
                pf_c1, pf_c2 = st.columns(2)
                full_name_pf = pf_c1.text_input(translate_text("Full Name", current_lang), value=p_fname or "")
                email_pf = pf_c2.text_input(translate_text("Email", current_lang), value=p_email or "")
                phone_pf = pf_c1.text_input(translate_text("Phone Number", current_lang), value=p_phone or "")
                address_pf = pf_c2.text_area(translate_text("Full Address", current_lang), value=p_address or "")
                region_pf = pf_c1.text_input(translate_text("Region/State", current_lang), value=p_region or "")
                
                st.caption(translate_text("Optional: Provide coordinates for better map visibility in 'Find a Farmer'. You can find these on Google Maps.", current_lang))
                lat_pf = pf_c2.number_input(translate_text("Farm Latitude (approx.)", current_lang), value=p_lat if p_lat else None, format="%.6f", help=translate_text("e.g., 18.520430", current_lang))
                lon_pf = pf_c1.number_input(translate_text("Farm Longitude (approx.)", current_lang), value=p_lon if p_lon else None, format="%.6f", help=translate_text("e.g., 73.856730", current_lang))
                
                share_contact_pf = st.checkbox(translate_text("Allow other users to see my Email/Phone on my listings?", current_lang), value=bool(p_share_contact), key="profile_share_contact")

                st.markdown("---")
                st.subheader(translate_text("Payment Information", current_lang))
                # New field for UPI ID
                upi_id_pf = st.text_input(
                    translate_text("UPI ID (e.g., yourname@bankname)", current_lang),
                    value=p_upi_id or "",
                    help=translate_text("Providing your UPI ID allows buyers to pay you directly for your products.", current_lang)
                )

                st.markdown("---")
                st.subheader(translate_text("🐄 Product Offerings", current_lang))
                sells_products_pf = st.checkbox(translate_text("I sell indigenous cow products from my farm", current_lang), value=bool(p_sells), key="profile_sells_prod")
                
                # Predefined list of common products for easier selection
                common_product_options = [
                    translate_text("Fresh A2 Milk", current_lang), 
                    translate_text("Desi Cow Ghee", current_lang), 
                    translate_text("Organic Manure (Dung)", current_lang), 
                    translate_text("Vermicompost", current_lang), 
                    translate_text("Gomutra (Cow Urine)", current_lang), 
                    translate_text("Panchagavya Mix", current_lang), 
                    translate_text("Dung Cakes (Uple/Kande)", current_lang), 
                    translate_text("Bio-Pesticides", current_lang), 
                    translate_text("Dhoop/Incense", current_lang)
                ]
                
                # Load current categories
                current_categories = p_prod_cat.split(',') if p_prod_cat else []
                current_categories = [cat.strip() for cat in current_categories if cat.strip()] # Clean up

                product_categories_pf = st.text_input(
                    translate_text("Products I Offer (comma-separated)", current_lang),
                    value=", ".join(current_categories), # Display as comma-separated string
                    help=translate_text("e.g., Milk, Ghee, Vermicompost, Dung Cakes", current_lang)
                )
                st.caption(translate_text(f"Example options: {', '.join(common_product_options)}", current_lang))


                submit_profile = st.form_submit_button(translate_text("💾 Save Profile Changes", current_lang))

                if submit_profile:
                    try:
                        # Convert product_categories_pf back to comma-separated string for DB
                        final_prod_cats = product_categories_pf.strip() if sells_products_pf else None

                        # Add 'upi_id' to the UPDATE query
                        cursor.execute("""UPDATE users SET
                                            full_name=?, email=?, phone_number=?, address=?, region=?,
                                            latitude=?, longitude=?, sells_products=?, product_categories=?, share_contact_info=?, upi_id=?
                                            WHERE user_id=? """,
                                        (full_name_pf or None, email_pf or None, phone_pf or None, address_pf or None, region_pf or None,
                                         lat_pf if lat_pf else None, lon_pf if lon_pf else None,
                                         1 if sells_products_pf else 0, final_prod_cats,
                                         1 if share_contact_pf else 0,
                                         upi_id_pf or None, # Save the UPI ID
                                         st.session_state.user_id))
                        conn.commit()
                        st.success(translate_text("Profile updated successfully!", current_lang))
                        st.rerun() # Refresh to show updated values
                    except sqlite3.IntegrityError: 
                        st.error(translate_text("Error: Email already in use by another account.", current_lang))
                    except sqlite3.Error as e_upd_prof: 
                        st.error(translate_text(f"Database error updating profile: {e_upd_prof}", current_lang))
        else:
            st.warning(translate_text("Could not load your profile data.", current_lang))

# Nutrition Planner Page (Farmer)

# Nutrition Planner Page (Farmer)
elif selected_page == "Nutrition Planner" and st.session_state.logged_in:
    ts.title("🥗 Nutrition Planner for Cattle")
    ts.markdown("Formulate and analyze rations for different life stages.")
    ts.markdown("---")

    conn = get_connection()
    if not conn:
        st.error("Database connection failed.")
    else:
        cursor = conn.cursor()

        # --- Animal Selection ---
        ts.subheader("1. Select Animal or Define Stage")
        col1_np, col2_np = st.columns(2)
        selected_animal_for_planning = None # Will store (animal_id, name, breed, sex, dob, weight_estimate_kg, current_status)

        with col1_np:
            use_my_herd = st.checkbox("Use animal from 'My Herd'?", value=True)
            if use_my_herd:
                cursor.execute("SELECT cattle_id, tag_id, name, breed, sex, dob, current_status FROM user_cattle WHERE user_id = ? AND current_status = 'Active' ORDER BY name, tag_id", (st.session_state.user_id,))
                my_active_cattle = cursor.fetchall()
                if my_active_cattle:
                    animal_options_planning = {f"{row[1] or 'No Tag'} - {row[2] or 'Unnamed'} (ID: {row[0]})": row[0] for row in my_active_cattle}
                    selected_animal_display = st.selectbox(translate_text("Select Your Animal",current_lang), options=list(animal_options_planning.keys()), index=0, key="np_animal_select")
                    if selected_animal_display:
                        selected_animal_id = animal_options_planning[selected_animal_display]
                        # Fetch more details if needed, here we assume the fetched details are enough for now
                        selected_animal_for_planning = next((animal for animal in my_active_cattle if animal[0] == selected_animal_id), None)
                        if selected_animal_for_planning:
                            ts.caption(f"Selected: {selected_animal_for_planning[2]} (Breed: {selected_animal_for_planning[3]}, Sex: {selected_animal_for_planning[4]})")
                else:
                    st.info(translate_text("No active cattle in 'My Herd'. Please add animals or define stage manually.",current_lang))
                    use_my_herd = False # Force manual input if no herd

        # Manual Animal Detail Input (if not using My Herd or My Herd is empty)
        animal_details_manual = {}
        if not use_my_herd or not selected_animal_for_planning:
            with col2_np:
                ts.write("Or, define animal stage manually:")
                animal_details_manual['life_stage'] = st.selectbox(translate_text("Life Stage",current_lang),
                                                                [translate_text("Lactating Cow",current_lang), translate_text("Dry Cow (Pregnant)",current_lang), translate_text("Heifer (Growing)",current_lang),
                                                                 translate_text("Calf (0-3 months)",current_lang), translate_text("Calf (3-6 months)",current_lang), translate_text("Bull (Breeding)",current_lang)], key="np_manual_stage")
                animal_details_manual['body_weight_kg'] = st.number_input(translate_text("Estimated Body Weight (kg)",current_lang), min_value=20, max_value=1000, value=400, step=10, key="np_manual_bw")
                if animal_details_manual['life_stage'] == "Lactating Cow":
                    animal_details_manual['milk_yield_l_day'] = st.number_input(translate_text("Daily Milk Yield (Liters)",current_lang), min_value=0.0, max_value=50.0, value=10.0, step=0.5, key="np_manual_milk")
                    animal_details_manual['lactation_stage'] = st.selectbox(translate_text("Lactation Stage",current_lang), [translate_text("Early (0-100 days)",current_lang), translate_text("Mid (101-200 days)",current_lang), translate_text("Late (>200 days)",current_lang)], key="np_manual_lact_stage")
                if "Pregnant" in animal_details_manual['life_stage']:
                     animal_details_manual['pregnancy_stage_months'] = st.slider(translate_text("Pregnancy Stage (Months into gestation)",current_lang), 1, 9, 7, key="np_manual_preg_stage")


        # --- 2. Nutrient Requirements (Placeholder Logic) ---
        # !!! THIS SECTION NEEDS TO BE REPLACED WITH ACCURATE DATA/FORMULAS (e.g., ICAR standards) !!!
        requirements = {"DM_kg": 0, "CP_g": 0, "ME_MJ": 0, "Ca_g": 0, "P_g": 0}
        animal_description_for_req = "Manually Defined Animal"

        def calculate_requirements(animal_data, manual_details):
            req = {"DM_kg": 0.0, "CP_g": 0.0, "ME_MJ": 0.0, "Ca_g": 0.0, "P_g": 0.0}
            desc = ""
            bw = 0

            if animal_data: # Using data from My Herd
                desc = f"{animal_data[2]} ({animal_data[3]}, {animal_data[4]})"
                # Estimate BW if not stored, or add a BW field to user_cattle
                # For now, a rough estimate based on sex/stage:
                if animal_data[4] in ['Cow', 'Female', 'Bull', 'Male']: bw = manual_details.get('body_weight_kg', 450) # Default or user input if available
                elif animal_data[4] == 'Heifer': bw = manual_details.get('body_weight_kg', 250)
                else: bw = manual_details.get('body_weight_kg', 100) # Calf

                # TODO: Refine logic based on user_cattle.sex, dob (to get age), and current_status for accurate stage
                # This is a very simplified placeholder
                if animal_data[4] == 'Cow' and 'Active' in animal_data[6]: # Assume lactating
                    stage = "Lactating Cow"
                    milk = manual_details.get('milk_yield_l_day', 10) # Get from manual input for now
                elif animal_data[4] == 'Heifer':
                    stage = "Heifer (Growing)"
                else:
                    stage = manual_details.get('life_stage', "Dry Cow (Pregnant)") # Default
            else: # Using manual details
                desc = f"Manually Defined: {manual_details.get('life_stage')}"
                bw = manual_details.get('body_weight_kg', 400)
                stage = manual_details.get('life_stage')
                milk = manual_details.get('milk_yield_l_day', 0) if stage == "Lactating Cow" else 0

            # Placeholder requirement calculations (VERY SIMPLIFIED - REPLACE WITH STANDARDS)
            req['DM_kg'] = bw * 0.025  # Roughly 2.5% of BW as DM
            if stage == "Lactating Cow":
                req['DM_kg'] += milk * 0.5 # Extra DM for milk
                req['CP_g'] = (bw * 1.0) + (milk * 70) # g/day CP (maintenance + production)
                req['ME_MJ'] = (bw * 0.3) + (milk * 5.0) # MJ/day ME (maintenance + production)
                req['Ca_g'] = (bw * 0.06) + (milk * 3.0)
                req['P_g'] = (bw * 0.04) + (milk * 2.0)
            elif stage == "Dry Cow (Pregnant)":
                preg_month = manual_details.get('pregnancy_stage_months', 7)
                factor = 1.1 if preg_month < 7 else 1.3 # Increase in late pregnancy
                req['CP_g'] = (bw * 1.0) * factor
                req['ME_MJ'] = (bw * 0.3) * factor
                req['Ca_g'] = (bw * 0.07) * factor
                req['P_g'] = (bw * 0.05) * factor
            elif stage == "Heifer (Growing)":
                req['CP_g'] = bw * 1.5 # Higher protein for growth
                req['ME_MJ'] = bw * 0.4
                req['Ca_g'] = bw * 0.08
                req['P_g'] = bw * 0.06
            elif "Calf" in stage:
                req['DM_kg'] = bw * 0.03 # Higher intake for calves
                req['CP_g'] = bw * 2.0 # Very high protein for young calves
                req['ME_MJ'] = bw * 0.5
                req['Ca_g'] = bw * 0.1
                req['P_g'] = bw * 0.08
            elif stage == "Bull (Breeding)":
                req['CP_g'] = bw * 1.2
                req['ME_MJ'] = bw * 0.35
                req['Ca_g'] = bw * 0.06
                req['P_g'] = bw * 0.04
            else: # Default maintenance if stage not matched
                req['CP_g'] = bw * 0.8
                req['ME_MJ'] = bw * 0.25
                req['Ca_g'] = bw * 0.05
                req['P_g'] = bw * 0.035

            # Ensure DM is not zero to avoid division by zero later
            if req['DM_kg'] <= 0 and bw > 0: req['DM_kg'] = bw * 0.02

            return req, desc, bw

        # Call calculation logic
        if selected_animal_for_planning or animal_details_manual.get('life_stage'):
            # If using 'My Herd', still allow overriding BW/Milk from manual fields if provided
            # This allows user to adjust if their stored herd data isn't detailed enough for planning
            effective_manual_details = animal_details_manual.copy()
            if selected_animal_for_planning:
                if 'body_weight_kg' not in effective_manual_details or effective_manual_details['body_weight_kg'] == 0 : # if not manually entered
                    # Try to get a default BW for "My Herd" animal, otherwise use a general default
                    animal_sex = selected_animal_for_planning[4]
                    if animal_sex in ['Cow', 'Female', 'Bull', 'Male']: default_bw_herd = 450
                    elif animal_sex == 'Heifer': default_bw_herd = 250
                    else: default_bw_herd = 100 # Calf
                    effective_manual_details['body_weight_kg'] = animal_details_manual.get('body_weight_kg',default_bw_herd)

                # If lactating, ensure milk yield is considered
                if selected_animal_for_planning[4] == 'Cow' and ('Lactating Cow' in animal_details_manual.get('life_stage', '') or not animal_details_manual.get('life_stage') ):
                    if 'milk_yield_l_day' not in effective_manual_details:
                         effective_manual_details['milk_yield_l_day'] = 10 # Default if not specified
                    if 'life_stage' not in effective_manual_details: # If not manually overridden
                         effective_manual_details['life_stage'] = 'Lactating Cow'

            requirements, animal_description_for_req, body_weight_used = calculate_requirements(selected_animal_for_planning, effective_manual_details)

            ts.subheader("2. Estimated Daily Nutrient Requirements")
            ts.write(f"For: **{animal_description_for_req}** (Body Weight: ~{body_weight_used:.0f} kg)")
            req_cols = st.columns(5)
            req_cols[0].metric("Dry Matter (DM)", f"{requirements['DM_kg']:.2f} kg")
            req_cols[1].metric("Crude Protein (CP)", f"{requirements['CP_g']:.0f} g")
            req_cols[2].metric("Metabolizable Energy (ME)", f"{requirements['ME_MJ']:.1f} MJ")
            req_cols[3].metric("Calcium (Ca)", f"{requirements['Ca_g']:.0f} g")
            req_cols[4].metric("Phosphorus (P)", f"{requirements['P_g']:.0f} g")
            st.caption(translate_text("Note: These are simplified estimates. Consult ICAR guidelines or a nutritionist for precise needs.",current_lang))
        else:
            st.info(translate_text("Please select an animal from 'My Herd' or define the animal stage manually above to see nutrient requirements.",current_lang))

        st.markdown("---")
        # --- 3. Formulate Ration ---
        ts.subheader("3. Formulate Daily Ration")

        # Fetch feedstuffs from DB
        try:
            cursor.execute("SELECT feedstuff_id, name, category, dm_percent, cp_percent, me_mj_kg_dm, ca_percent, p_percent FROM feedstuffs ORDER BY category, name")
            all_feedstuffs_db = cursor.fetchall()
            feedstuff_options = {f"{row[1]} ({row[2]})": row for row in all_feedstuffs_db} # Name: (id, name, cat, dm, cp, me, ca, p)
        except sqlite3.Error as e:
            st.error(translate_text(f"Could not load feedstuffs: {e}",current_lang))
            all_feedstuffs_db = []
            feedstuff_options = {}

        if 'ration_items' not in st.session_state:
            st.session_state.ration_items = [] # List of dicts: {'feed_name': str, 'feed_data': tuple, 'as_fed_kg': float}

        def add_ration_item():
            st.session_state.ration_items.append({'feed_name': None, 'feed_data': None, 'as_fed_kg': 1.0})

        def remove_ration_item(index):
            if 0 <= index < len(st.session_state.ration_items):
                st.session_state.ration_items.pop(index)

        # Display ration items
        form_cols = st.columns([3, 2, 1]) # Feedstuff, Amount, Remove
        for i, item in enumerate(st.session_state.ration_items):
            with st.container(): # Keep each item on its own row visually
                 cols = st.columns([4,2,1])
                 selected_feed_name = cols[0].selectbox(f"Feedstuff {i+1}", ["Select Feedstuff..."] + list(feedstuff_options.keys()),
                                                      key=f"feed_sel_{i}",
                                                      index= (list(feedstuff_options.keys()).index(item['feed_name']) + 1) if item['feed_name'] else 0)
                 if selected_feed_name != "Select Feedstuff...":
                     st.session_state.ration_items[i]['feed_name'] = selected_feed_name
                     st.session_state.ration_items[i]['feed_data'] = feedstuff_options[selected_feed_name]
                 else:
                     st.session_state.ration_items[i]['feed_name'] = None
                     st.session_state.ration_items[i]['feed_data'] = None


                 st.session_state.ration_items[i]['as_fed_kg'] = cols[1].number_input("As-Fed (kg)", min_value=0.0, value=item['as_fed_kg'], step=0.1, key=f"feed_kg_{i}")
                 if cols[2].button("🗑️", key=f"del_feed_{i}", help="Remove this feedstuff"):
                     remove_ration_item(i)
                     st.experimental_rerun() # Rerun to reflect removal

        if st.button(translate_text("➕ Add Feedstuff to Ration",current_lang, key="auto_btn_9"), on_click=add_ration_item):
            pass # Action handled by on_click

        st.markdown("---")
        if st.button(translate_text("📊 Analyze Formulated Ration",current_lang, key="auto_btn_10"), type="primary"):
            if not any(item['feed_data'] for item in st.session_state.ration_items):
                st.warning(translate_text("Please add at least one feedstuff to the ration and select it.",current_lang))
            elif requirements["DM_kg"] == 0: # Check if requirements were calculated
                 st.warning(translate_text("Please select an animal or define its stage first to calculate requirements.",current_lang))
            else:
                total_supplied = {"DM_kg": 0.0, "CP_g": 0.0, "ME_MJ": 0.0, "Ca_g": 0.0, "P_g": 0.0}
                ration_summary_data = []

                for item in st.session_state.ration_items:
                    if item['feed_data']:
                        feed_id, name, cat, dm_pct, cp_pct, me_mj, ca_pct, p_pct = item['feed_data']
                        as_fed = item['as_fed_kg']
                        if dm_pct is None or cp_pct is None or me_mj is None or ca_pct is None or p_pct is None:
                            st.warning(translate_text(f"Nutrient data missing for {name}. It will be excluded from totals.",current_lang), icon="⚠️")
                            continue

                        dm_supplied = as_fed * (dm_pct / 100.0)
                        cp_supplied_g = dm_supplied * (cp_pct / 100.0) * 1000 # CP is % of DM, convert to g
                        me_supplied_mj = dm_supplied * me_mj
                        ca_supplied_g = dm_supplied * (ca_pct / 100.0) * 1000
                        p_supplied_g = dm_supplied * (p_pct / 100.0) * 1000

                        total_supplied["DM_kg"] += dm_supplied
                        total_supplied["CP_g"] += cp_supplied_g
                        total_supplied["ME_MJ"] += me_supplied_mj
                        total_supplied["Ca_g"] += ca_supplied_g
                        total_supplied["P_g"] += p_supplied_g
                        ration_summary_data.append([name, f"{as_fed:.2f}", f"{dm_supplied:.2f}", f"{cp_supplied_g:.0f}", f"{me_supplied_mj:.1f}", f"{ca_supplied_g:.1f}", f"{p_supplied_g:.1f}"])

                ts.subheader("4. Ration Analysis")
                if ration_summary_data:
                    df_ration = pd.DataFrame(ration_summary_data, columns=["Feedstuff", "As-Fed (kg)", "DM (kg)", "CP (g)", "ME (MJ)", "Ca (g)", "P (g)"])
                    st.dataframe(df_ration, hide_index=True, use_container_width=True)

                ts.markdown("**Comparison: Supplied vs. Required**")
                analysis_cols = st.columns(5)
                nutrients_to_compare = [
                    ("DM", "DM_kg", "kg", ".2f"),
                    ("CP", "CP_g", "g", ".0f"),
                    ("ME", "ME_MJ", "MJ", ".1f"),
                    ("Ca", "Ca_g", "g", ".0f"),
                    ("P", "P_g", "g", ".0f")
                ]
                for i, (label, key, unit, fmt) in enumerate(nutrients_to_compare):
                    supplied_val = total_supplied[key]
                    required_val = requirements[key]
                    balance = supplied_val - required_val
                    balance_pct = (balance / required_val * 100) if required_val > 0 else 0

                    analysis_cols[i].metric(
                        label=f"{label} ({unit})",
                        value=f"{supplied_val:{fmt}}",
                        delta=f"{balance:{fmt}} ({balance_pct:+.0f}%)",
                        delta_color="normal" if abs(balance_pct) <= 10 else ("inverse" if balance < 0 else "normal") # Greenish if surplus, Reddish if deficit >10%
                    )
                    # analysis_cols[i].write(f"Required: {required_val:{fmt}} {unit}")

elif selected_page == "Sell Cattle" and st.session_state.logged_in and st.session_state.role == "farmer":
    ts.title("🛒 List Your Cattle for Sale")
    ts.markdown("Offer your animals to interested buyers directly.")
    st.markdown("---")
    conn = get_connection()
    if not conn: st.error("Database connection failed.")
    else:
        cursor = conn.cursor()
        ts.subheader("1. Select Animal to List for Sale")
        cursor.execute("""
            SELECT uc.cattle_id, uc.tag_id, uc.name, uc.breed
            FROM user_cattle uc
            LEFT JOIN cattle_for_sale cfs ON uc.cattle_id = cfs.cattle_id
            WHERE uc.user_id = ? AND uc.current_status = 'Active' AND cfs.listing_id IS NULL
            ORDER BY uc.name, uc.tag_id
        """, (st.session_state.user_id,))
        available_to_sell = cursor.fetchall()

        if not available_to_sell:
            st.info(translate_text("You have no 'Active' cattle in 'My Herd' that are not already listed for sale. Add animals to 'My Herd' first or update their status.",current_lang))
        else:
            sell_options = {f"{row[1] or 'No Tag'} - {row[2] or 'Unnamed'} (Breed: {row[3] or 'N/A'}, ID: {row[0]})": row[0] for row in available_to_sell}
            selected_animal_to_sell_display = st.selectbox("Choose an animal from your herd:",
                                                           options=["--- Select Animal ---"] + list(sell_options.keys()),
                                                           index=0, key="sell_animal_select")
            if selected_animal_to_sell_display != "--- Select Animal ---":
                selected_cattle_id_to_sell = sell_options[selected_animal_to_sell_display]
                ts.caption(f"You are about to list: {selected_animal_to_sell_display}")

                with st.form(translate_text("sell_cattle_form",current_lang), clear_on_submit=True): # Clear form after successful submission
                    ts.subheader("2. Listing Details")
                    asking_price_sc = ts.number_input("Asking Price (₹)*", min_value=1000.0, step=500.0, key="sell_price_cattle")
                    description_sc = ts.text_area("Description*", height=150, key="sell_desc_cattle", placeholder="e.g., temperament, milking history, any special traits, age, lactation number...")
                    location_sell_sc = ts.text_input("Your Location* (e.g., District, State)", key="sell_location_cattle", placeholder="Buyers will see this location")
                    
                    ts.markdown("##### Upload Images (Optional, up to 2)")
                    img_col1_sc, img_col2_sc = st.columns(2)
                    with img_col1_sc:
                        image1_cattle_sc = st.file_uploader(translate_text("Image 1 (Main Image)",current_lang), type=['jpg', 'jpeg', 'png'], key="sell_cattle_img1")
                    with img_col2_sc:
                        image2_cattle_sc = st.file_uploader(translate_text("Image 2 (Optional)",current_lang), type=['jpg', 'jpeg', 'png'], key="sell_cattle_img2")

                    submit_listing_sc = st.form_submit_button(translate_text("✅ List This Animal for Sale",current_lang))

                    if submit_listing_sc:
                        if asking_price_sc <= 0 or not description_sc.strip() or not location_sell_sc.strip():
                             st.error(translate_text("Please fill in Asking Price, Description, and Location.",current_lang))
                        else:
                            image_path_1_sc = save_uploaded_image(image1_cattle_sc, "cattle") if image1_cattle_sc else None
                            image_path_2_sc = save_uploaded_image(image2_cattle_sc, "cattle") if image2_cattle_sc else None
                            try:
                                cursor.execute("""
                                    INSERT INTO cattle_for_sale (user_id, cattle_id, asking_price, description, location, status, image_url_1, image_url_2)
                                    VALUES (?, ?, ?, ?, ?, 'Available', ?, ?)
                                """, (st.session_state.user_id, selected_cattle_id_to_sell, asking_price_sc, description_sc, location_sell_sc, image_path_1_sc, image_path_2_sc))
                                cursor.execute("UPDATE user_cattle SET current_status = 'For Sale' WHERE cattle_id = ? AND user_id = ?",
                                               (selected_cattle_id_to_sell, st.session_state.user_id))
                                conn.commit()
                                st.success(translate_text(f"Animal '{selected_animal_to_sell_display}' has been successfully listed for sale!",current_lang))
                                logger.info(f"Farmer {st.session_state.username} listed cattle ID {selected_cattle_id_to_sell} for sale.")
                                # st.rerun() # Rerun can sometimes clear the success message too quickly, let user see it.
                            except sqlite3.IntegrityError:
                                st.error(translate_text("This animal might already be listed or there's a data issue. Please check.",current_lang))
                            except sqlite3.Error as e:
                                st.error(f"Database error: {e}")
                                logger.error(f"DB error listing cattle for {st.session_state.username}: {e}")
        st.markdown("---")
        ts.subheader("🗓️ Your Active Sale Listings (Cattle)")
        try:
            cursor.execute("""
                SELECT cfs.listing_id, uc.tag_id, uc.name, uc.breed, cfs.asking_price, cfs.listing_date, cfs.status, cfs.image_url_1
                FROM cattle_for_sale cfs
                JOIN user_cattle uc ON cfs.cattle_id = uc.cattle_id
                WHERE cfs.user_id = ? AND cfs.status = 'Available'
                ORDER BY cfs.listing_date DESC
            """, (st.session_state.user_id,))
            active_listings_sc = cursor.fetchall()
            if active_listings_sc:
                for listing_data_sc in active_listings_sc:
                    list_id_sc, tag_sc, name_sc, breed_sc, price_sc, listed_on_sc, status_sc, img_path_sc = listing_data_sc
                    with st.container(border=True):
                        disp_c1_sc, disp_c2_sc = st.columns([1, 3])
                        with disp_c1_sc:
                            display_uploaded_image(img_path_sc, caption="Main Image", width=120)
                        with disp_c2_sc:
                            ts.subheader(f"{name_sc or 'Unnamed'} (Tag: {tag_sc or 'N/A'})")
                            ts.markdown(f"**Breed:** {breed_sc} | **Asking Price:** ₹{price_sc:,.0f}")
                            try:
                                listed_date_str = datetime.strptime(listed_on_sc, '%Y-%m-%d %H:%M:%S').strftime('%d %b %Y')
                            except ValueError: # Handle potential old format if any
                                listed_date_str = listed_on_sc
                            ts.caption(f"Listed: {listed_date_str} | Status: {status_sc} | Listing ID: {list_id_sc}")
                            # TODO: Add button to "View Details" / "Edit" / "Withdraw" listing
                            if st.button("Withdraw Listing", key=f"withdraw_cattle_{list_id_sc}", type="secondary"):
                                try:
                                    cursor.execute("UPDATE cattle_for_sale SET status = 'Withdrawn' WHERE listing_id = ? AND user_id = ?", (list_id_sc, st.session_state.user_id))
                                    cursor.execute("UPDATE user_cattle SET current_status = 'Active' WHERE cattle_id = (SELECT cattle_id FROM cattle_for_sale WHERE listing_id = ?) AND user_id = ?", (list_id_sc, st.session_state.user_id))
                                    conn.commit()
                                    st.success(f"Listing ID {list_id_sc} withdrawn.")
                                    st.rerun()
                                except sqlite3.Error as e_wd:
                                    st.error(f"Error withdrawing listing: {e_wd}")
                        st.markdown("---") # Separator inside the loop for each listing
            else:
                st.info(translate_text("You have no active cattle listings.",current_lang))
        except sqlite3.Error as e:
            st.error(translate_text(f"Could not fetch your sale listings: {e}",current_lang))

# Your existing dashboard code, with the connection fix
if selected_page == "Farmer Dashboard" and st.session_state.logged_in and st.session_state.role == "farmer":
    ts.title(f"🧑‍🌾 Farmer Dashboard - Welcome {st.session_state.username}!")
    ts.markdown("Your central hub for farm management and local insights.")
    st.markdown("---")

    # --- FIX START ---
    conn = get_connection()
    if not conn:
        st.error("Database connection failed. Please check the database file or logs for errors.")
        st.stop()
    cursor = conn.cursor()
    # --- FIX END ---

    # --- Quick Stats & Links ---
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    try:
        cursor.execute("SELECT COUNT(*) FROM user_cattle WHERE user_id = ? AND current_status = 'Active'", (st.session_state.user_id,))
        active_cattle_count = cursor.fetchone()[0]
        col_stat1.metric(translate_text("Active Cattle in Herd",current_lang), active_cattle_count)
    except Exception as e: col_stat1.caption(f"Error: {e}")
    try:
        cursor.execute("SELECT COUNT(*) FROM cattle_for_sale WHERE user_id = ? AND status = 'Available'", (st.session_state.user_id,))
        cattle_for_sale_count = cursor.fetchone()[0]
        col_stat2.metric(translate_text("Cattle Listed for Sale",current_lang), cattle_for_sale_count)
    except Exception as e: col_stat2.caption(f"Error: {e}")
    try:
        cursor.execute("SELECT COUNT(*) FROM machinery_listings WHERE user_id = ? AND status = 'Available'", (st.session_state.user_id,))
        mach_for_sale_count = cursor.fetchone()[0]
        col_stat3.metric(translate_text("Machinery For Sale/Rent",current_lang), mach_for_sale_count)
    except Exception as e: col_stat3.caption(f"Error: {e}")

    ts.markdown("##### Quick Actions:")
    q_cols = st.columns(5)
    if q_cols[0].button(translate_text("📋 My Herd",current_lang), use_container_width=True, key="dash_herd_btn"): st.session_state.current_page = "My Herd"; st.rerun()
    if q_cols[1].button(translate_text("🔔 My Profile",current_lang), use_container_width=True, key="dash_profile_settings_btn"): st.session_state.current_page = "My Profile"; st.rerun()
    if q_cols[2].button(translate_text("🛒 Sell Cattle",current_lang), use_container_width=True, key="dash_sell_c_btn"): st.session_state.current_page = "Sell Cattle"; st.rerun()
    if q_cols[3].button(translate_text("🚜 Sell Machinery",current_lang), use_container_width=True, key="dash_sell_m_btn"): st.session_state.current_page = "Sell Machinery"; st.rerun()
    if q_cols[4].button(translate_text("🔍 Browse Machinery",current_lang), use_container_width=True, key="dash_browse_m_btn"): st.session_state.current_page = "Browse Machinery"; st.rerun()
    st.markdown("---")

    # --- Health & Breeding Reminders Section ---
    ts.subheader("🔔 Upcoming Health & Breeding Reminders (Next 5)")
    alerts_col, breeding_reminders_col = st.columns(2)

    with alerts_col:
        ts.markdown("##### Health Actions:")
        due_health_reminders_dashboard = []
        try:
            cursor.execute("""
                SELECT hrs.reminder_description, hrs.due_date, uc.name as cattle_name, uc.tag_id
                FROM health_reminders_status hrs
                JOIN user_cattle uc ON hrs.cattle_id = uc.cattle_id
                WHERE hrs.user_id = ? AND hrs.status = 'pending' AND date(hrs.due_date) >= date('now')
                ORDER BY hrs.due_date ASC
                LIMIT 3 
            """, (st.session_state.user_id,))
            due_health_reminders_dashboard = cursor.fetchall()
            if due_health_reminders_dashboard:
                for r_desc, r_due, c_name, c_tag in due_health_reminders_dashboard:
                    icon = "💉" if "vaccination" in r_desc.lower() or "booster" in r_desc.lower() else "🩺"
                    due_date_obj = datetime.strptime(r_due, '%Y-%m-%d').date()
                    if due_date_obj == date.today():
                        display_text = f"{icon} *TODAY:* {r_desc} for {c_name or c_tag or 'Animal'}"
                        st.error(display_text,icon="🚨")
                    else:
                        display_text = f"{icon} {r_desc} for {c_name or c_tag or 'Animal'} on {due_date_obj.strftime('%d %b %Y')}"
                        st.warning(display_text,icon="🗓")
            else:
                ts.info("No upcoming health actions noted as pending.")
        except sqlite3.Error as e:
            ts.info(f"Reminder data may be unavailable: {e}. Ensure health_reminders_status table exists.")

    with breeding_reminders_col:
        ts.markdown("##### Breeding Cycle:")
        breeding_alerts = []
        try:
            cursor.execute("""
                SELECT cattle_id, name, tag_id, expected_due_date, last_insemination_date, pregnancy_status, last_calving_date, sex
                FROM user_cattle
                WHERE user_id = ? AND current_status = 'Active'
            """, (st.session_state.user_id,))
            breeding_cattle = cursor.fetchall()
            today_date = date.today()
            breeding_alerts_count = 0
            for c_id, c_name, c_tag, edd_str, lid_str, preg_stat, lcd_str, sex_animal in breeding_cattle:
                if breeding_alerts_count >= 2: break
                animal_display = f"{c_name or c_tag or f'Animal ID {c_id}'}"
                if edd_str:
                    edd_date = datetime.strptime(edd_str, '%Y-%m-%d').date()
                    if today_date <= edd_date <= (today_date + timedelta(days=30)):
                        breeding_alerts.append(f"🤰 *Expected Calving:* {animal_display} around {edd_date.strftime('%d %b %Y')}")
                        breeding_alerts_count +=1
                if lid_str and preg_stat == "Inseminated - Awaiting Check" and breeding_alerts_count < 2:
                    lid_date = datetime.strptime(lid_str, '%Y-%m-%d').date()
                    pd_due_date_min = lid_date + timedelta(days=45) 
                    pd_due_date_max = lid_date + timedelta(days=75) 
                    if pd_due_date_min <= today_date <= pd_due_date_max:
                        breeding_alerts.append(f"🔬 *Pregnancy Check Due:* {animal_display} (Insem. {lid_date.strftime('%d %b')})")
                        breeding_alerts_count +=1
                    elif today_date > pd_due_date_max : 
                        breeding_alerts.append(f"⚠ *Overdue Preg. Check:* {animal_display} (Insem. {lid_date.strftime('%d %b')})")
                        breeding_alerts_count +=1
                if sex_animal in ['Female', 'Cow', 'Heifer'] and lcd_str and \
                   (preg_stat is None or preg_stat == "Open" or preg_stat == "Recently Calved") and \
                   breeding_alerts_count < 2:
                    lcd_date = datetime.strptime(lcd_str, '%Y-%m-%d').date()
                    heat_check_due_start = lcd_date + timedelta(days=40)
                    heat_check_due_end = lcd_date + timedelta(days=75)
                    if heat_check_due_start <= today_date <= heat_check_due_end:
                        breeding_alerts.append(f"🔥 *Observe for Heat:* {animal_display} (Calved {lcd_date.strftime('%d %b')})")
                        breeding_alerts_count +=1
            if breeding_alerts:
                for alert_item in breeding_alerts:
                    ts.info(alert_item, icon="🐄")
            else:
                ts.info("No immediate breeding cycle events noted.",icon="ℹ️")
        except sqlite3.Error as e:
            st.warning(f"Error fetching breeding reminders: {e}")
    st.markdown("---")

    # --- Prosperity Board Section (New) ---
    ts.subheader("💰 Prosperity Board: Your Farm's Financial Health")
    st.write(translate_text("Gain insights into your farm's income, expenses, and overall economic performance.", current_lang))

    # Example placeholders for Prosperity Board content:
    pb_col1, pb_col2 = st.columns(2)
    with pb_col1:
        ts.markdown("##### 📈 Income & Expenses (Last 30 Days)")
        try:
            # You'd need actual tables for 'income_records' and 'expense_records'
            # For demonstration, let's use dummy values
            total_income = 150000 # Example data
            total_expenses = 75000 # Example data
            
            st.metric(translate_text("Total Income", current_lang), f"₹ {total_income:,.2f}")
            st.metric(translate_text("Total Expenses", current_lang), f"₹ {total_expenses:,.2f}")
            st.metric(translate_text("Net Profit", current_lang), f"₹ {total_income - total_expenses:,.2f}")
        except Exception as e:
            ts.caption(f"Error loading financial data: {e}")

    with pb_col2:
        ts.markdown("##### 🚀 Key Performance Indicators")
        try:
            # Again, these would come from your DB, or calculations based on it
            avg_milk_yield = 12.5 # Example L/day per active cow
            calving_rate = "85%" # Example
            
            st.metric(translate_text("Average Milk Yield", current_lang), f"{avg_milk_yield} L/day")
            st.metric(translate_text("Successful Calving Rate", current_lang), calving_rate)
            # You could add charts here if you have enough data:
            # st.line_chart(pd.DataFrame({'Month': ['Jan', 'Feb', 'Mar'], 'Yield': [10, 11, 12]}))
        except Exception as e:
            ts.caption(f"Error loading KPI data: {e}")
    
    ts.markdown("##### 🏛️ Government Schemes & Resources")
    st.info(translate_text("Explore available government subsidies and schemes for cattle farming. Check the 'Schemes' section for more.", current_lang))
    if st.button(translate_text("View Government Schemes", current_lang, key="auto_btn_12"), key="view_schemes_pb"):
        st.session_state.current_page = "Govt Schemes" # Assuming you have a page for this
        st.rerun()

    st.markdown("---")

    # Weather Advisory Section
    ts.subheader("🌦 Local Weather Advisory & Cattle Care Tips")
    user_region = "Nagpur" 
    try:
        cursor.execute("SELECT region FROM users WHERE user_id = ?", (st.session_state.user_id,))
        db_region = cursor.fetchone()
        if db_region and db_region[0]:
            user_region = db_region[0].split(',')[0].strip() 
    except: pass 

    available_cities = ["Nagpur", "Delhi", "Mumbai", "Bengaluru", "Jaipur", "Bhopal", "Patna", "Pune", "Hyderabad", "Chennai", "Kolkata", "Lucknow", "Chandigarh", "Ahmedabad"] 
    selected_city_index = 0 
    normalized_user_region = user_region.lower()
    for i, city in enumerate(available_cities):
        if city.lower() == normalized_user_region:
            selected_city_index = i
            break
    selected_city_for_weather = st.selectbox(translate_text("Select Your Nearest Major City for Weather (or update in profile):",current_lang),available_cities, index=selected_city_index, key="farmer_dashboard_weather_city_select")
    if selected_city_for_weather:
        coords = get_coordinates(selected_city_for_weather)
        if coords:
            weather_data = fetch_weather_forecast(coords['latitude'], coords['longitude'])
            if weather_data and 'daily' in weather_data and 'time' in weather_data['daily']:
                daily_df_data = {
                    "Date": [datetime.strptime(d, "%Y-%m-%d").strftime("%a, %d %b") for d in weather_data['daily']['time']],
                    "Max Temp (°C)": weather_data['daily']['temperature_2m_max'],
                    "Min Temp (°C)": weather_data['daily']['temperature_2m_min'],
                    "Precip. (mm)": weather_data['daily']['precipitation_sum'],
                    "Precip. Prob (%)": weather_data['daily']['precipitation_probability_max'],
                    "Outlook": [interpret_weather_code(code) for code in weather_data['daily']['weather_code']]
                }
                forecast_df = pd.DataFrame(daily_df_data)
                with st.expander(translate_text("📅 7-Day Weather Forecast Details",current_lang), expanded=False):
                    st.dataframe(forecast_df.set_index("Date"), use_container_width=True)
                
                today_forecast_data = {key: weather_data['daily'][key][0] for key in weather_data['daily']} if len(weather_data['daily']['time']) > 0 else None
                next_days_forecast_data = [{key: weather_data['daily'][key][i] for key in weather_data['daily']} for i in range(1, min(len(weather_data['daily']['time']), 7))]
                
                advice, notifications = generate_cattle_care_advice(today_forecast_data, next_days_forecast_data)
                
                col_notify, col_advice_disp = st.columns([1,2])
                with col_notify: 
                    ts.markdown("##### ⚠ Notifications:")
                    if notifications: 
                        for notification in notifications: st.info(notification)
                    else: ts.caption("No critical notifications.")
                with col_advice_disp: 
                    ts.markdown("##### 💡 Care Suggestions:")
                    if advice: 
                        for item in advice: st.markdown(translate_text(f"{item}",current_lang))
                    else: ts.caption("No specific advice.")
            else: ts.caption("Weather data unavailable or incomplete.")
        else: st.error(translate_text(f"Coordinates not found for {selected_city_for_weather}.",current_lang))
    st.markdown("---")

    # 📊 Market Insights Section
    ts.subheader("📊 Cattle Market Insights")
    try:
        market_trends = [
            {translate_text("Breed",current_lang): translate_text("Gir",current_lang),translate_text( "Avg Price (₹)",current_lang): 55000, translate_text("Demand",current_lang): translate_text("High",current_lang)},
            {translate_text("Breed",current_lang): translate_text("Sahiwal",current_lang), translate_text("Avg Price (₹)",current_lang): 48000, translate_text("Demand",current_lang): translate_text("Moderate",current_lang)},
            {translate_text("Breed",current_lang): translate_text("Jersey",current_lang), translate_text("Avg Price (₹)",current_lang): 45000, translate_text("Demand",current_lang): translate_text("Low",current_lang)},
        ]
        market_df = pd.DataFrame(market_trends)
        st.dataframe(market_df, use_container_width=True)
        ts.caption("📈 Prices are indicative and based on recent listings and buyer demand trends.")
    except Exception as e:
        st.error(translate_text(f"Failed to load market insights: {e}",current_lang))

    st.markdown("---")

    # 📰 Agri-News Feed (static mock for now)
    ts.subheader("📰 Agri-News & Updates")
    news_items = [
        "🌱 Govt. announces new subsidy for organic cattle feed.",
        "🚚 Digital livestock marketplace launched in Maharashtra.",
        "🩺 New vaccination guidelines issued for foot-and-mouth disease.",
    ]
    for item in news_items:
        ts.write(f"- {item}")



# NEW: Browse Cattle for Sale Page (Buyer)
elif st.session_state.current_page == "Browse Cattle":  # Assuming access control is done before this block
    ts.title("🔍 " + translate_text("Browse Cattle for Sale", current_lang))
    ts.markdown(translate_text("Find cattle listed directly by farmers.", current_lang))
    st.markdown("---")
    conn = get_connection()
    if not conn:
        st.error(translate_text("Database connection failed.", current_lang))
    else:
        cursor = conn.cursor()

        # --- Search and Filters ---
        s_c1_bc, s_c2_bc, s_c3_bc = st.columns(3)
        search_breed_bc = s_c1_bc.text_input(translate_text("Search by Breed", current_lang), key="buy_search_breed_bc")
        search_location_bc = s_c2_bc.text_input(translate_text("Search by Location (District/State)", current_lang), key="buy_search_loc_bc")
        max_price_bc = s_c3_bc.number_input(translate_text("Max Price (₹)", current_lang), min_value=0, value=0, step=1000, key="buy_max_price_bc", help=translate_text("0 for no limit", current_lang))

        # Updated query to fetch all necessary details, including UPI ID directly
        query_bc = """
            SELECT
                cfs.listing_id,
                u.username as seller_username,
                u_seller_profile.full_name as seller_full_name,
                u_seller_profile.region as seller_region,
                u_seller_profile.email as seller_email,
                u_seller_profile.phone_number as seller_phone,
                u_seller_profile.share_contact_info,
                u_seller_profile.upi_id, -- Added UPI ID here
                uc.tag_id,
                uc.name as animal_name,
                uc.breed,
                uc.sex,
                uc.dob,
                uc.notes as animal_notes,
                uc.sire_tag_id,
                uc.dam_tag_id,
                uc.purchase_date,
                uc.purchase_price,
                cfs.asking_price,
                cfs.description as listing_description,
                cfs.location as listing_location,
                cfs.listing_date,
                cfs.image_url_1,
                cfs.image_url_2
            FROM cattle_for_sale cfs
            JOIN user_cattle uc ON cfs.cattle_id = uc.cattle_id
            JOIN users u ON cfs.user_id = u.user_id
            JOIN users u_seller_profile ON cfs.user_id = u_seller_profile.user_id
            WHERE cfs.status = 'Available'
        """
        params_bc = []
        if search_breed_bc:
            query_bc += " AND LOWER(uc.breed) LIKE ?"
            params_bc.append(f"%{search_breed_bc.lower()}%")
        if search_location_bc:
            query_bc += " AND LOWER(cfs.location) LIKE ?"
            params_bc.append(f"%{search_location_bc.lower()}%")
        if max_price_bc > 0:
            query_bc += " AND cfs.asking_price <= ?"
            params_bc.append(max_price_bc)

        query_bc += " ORDER BY cfs.listing_date DESC"

        try:
            cursor.execute(query_bc, params_bc)
            listings_bc_data = cursor.fetchall()
            if listings_bc_data:
                st.subheader(translate_text(f"Found {len(listings_bc_data)} cattle for sale:", current_lang))
                for listing_tuple_bc in listings_bc_data:
                    (listing_id, seller_username, seller_full_name, seller_region, seller_email, seller_phone, seller_share_contact,
                     seller_upi_id, # Now directly available in the tuple
                     tag_id, animal_name, breed, sex, dob, animal_notes, sire_tag_id, dam_tag_id, purchase_date, purchase_price,
                     asking_price, listing_description, listing_location, listing_date_str,
                     image_url_1, image_url_2) = listing_tuple_bc
                    
                    is_already_saved_cattle = False
                    if st.session_state.logged_in:
                        try:
                            cursor.execute("""SELECT saved_id FROM user_saved_listings
                                            WHERE user_id = ? AND listing_type = 'cattle' AND original_listing_id = ?""",
                                           (st.session_state.user_id, listing_id))
                            if cursor.fetchone():
                                is_already_saved_cattle = True
                        except sqlite3.Error as e_check_save:
                            logger.error(f"Error checking saved cattle {listing_id}: {e_check_save}")

                    with st.container(border=True):
                        col_summary_img_bc, col_summary_info_bc = st.columns([1, 3])
                        with col_summary_img_bc:
                            display_uploaded_image(image_url_1, caption=animal_name, use_container_width=True)
                        with col_summary_info_bc:
                            st.subheader(f"{animal_name or translate_text('Unnamed Animal', current_lang)} (Tag: {tag_id or 'N/A'})")
                            age_str_bc = "N/A"
                            if dob:
                                try:
                                    birth_date_bc = datetime.strptime(dob, "%Y-%m-%d").date()
                                    today_bc = date.today()
                                    age_delta_bc = today_bc - birth_date_bc
                                    years_bc = age_delta_bc.days // 365
                                    months_bc = (age_delta_bc.days % 365) // 30
                                    age_str_bc = f"{years_bc}y {months_bc}m"
                                except:
                                    pass
                            st.markdown(translate_text(f"**Breed:** {breed or 'N/A'} | **Sex:** {sex or 'N/A'} | **Age:** {age_str_bc}", current_lang))
                            st.markdown(translate_text(f"**Location:** {listing_location or 'N/A'} | **Price:** ₹{asking_price:,.0f}", current_lang))
                            st.caption(translate_text(f"Listed by: {seller_username} on {datetime.strptime(listing_date_str, '%Y-%m-%d %H:%M:%S').strftime('%d %b %Y')}", current_lang))

                            is_this_cattle_expanded = (st.session_state.expanded_cattle_listing_id == listing_id)
                            button_label = translate_text("➖ Hide Details", current_lang) if is_this_cattle_expanded else translate_text("🔍 View Full Details & Contact", current_lang)

                            if st.button(button_label, key=f"view_cattle_expander_{listing_id}"):
                                if is_this_cattle_expanded:
                                    st.session_state.expanded_cattle_listing_id = None  # Collapse
                                else:
                                    st.session_state.expanded_cattle_listing_id = listing_id  # Expand
                                    st.session_state.expanded_machinery_listing_id = None  # Collapse others
                                st.rerun()

                        if is_this_cattle_expanded:
                            st.markdown("---")
                            st.markdown(translate_text(f"##### Full Details for: {animal_name or 'Unnamed'}", current_lang))
                            st.markdown(translate_text(f"**Listing ID:** {listing_id}", current_lang))
                            st.markdown(translate_text(f"**Seller's Description:**\n{listing_description or 'No further description provided.'}", current_lang))
                            if animal_notes:
                                st.markdown(translate_text(f"**Breeder's Notes (from herd record):** {animal_notes}", current_lang))
                            if sire_tag_id:
                                st.markdown(translate_text(f"**Sire (Tag ID):** {sire_tag_id}", current_lang))
                            if dam_tag_id:
                                st.markdown(translate_text(f"**Dam (Tag ID):** {dam_tag_id}", current_lang))
                            if dob:
                                st.markdown(translate_text(f"**Date of Birth:** {dob}", current_lang))
                            if purchase_date:
                                st.markdown(translate_text(f"**Original Purchase Date:** {purchase_date}", current_lang))
                            if purchase_price:
                                st.markdown(translate_text(f"**Original Purchase Price:** ₹{purchase_price:,.0f}", current_lang))

                            if image_url_2:
                                st.markdown(translate_text("**Additional Image:**", current_lang))
                                display_uploaded_image(image_url_2, caption=translate_text("Additional Image", current_lang), use_container_width=True)
                            st.markdown("---")
                            st.subheader(translate_text(f"Contact Seller: {seller_username}", current_lang))
                            st.markdown(translate_text(f"**Name:** {seller_full_name or seller_username}", current_lang))
                            st.markdown(translate_text(f"**Region:** {seller_region or 'N/A'}", current_lang))
                            if seller_share_contact == 1:
                                st.markdown(translate_text("**Contact Information:**", current_lang))
                                if seller_email:
                                    st.markdown(translate_text(f"📧 Email: [{seller_email}](mailto:{seller_email}?subject=Inquiry about Cattle: {animal_name or tag_id} - Listing ID {listing_id})", current_lang))
                                if seller_phone:
                                    st.markdown(translate_text(f"📞 Phone: `{seller_phone}`", current_lang))
                                st.caption(translate_text("Please be respectful when contacting sellers.", current_lang))
                            else:
                                st.info(translate_text("Seller has chosen not to share direct contact details publicly.", current_lang))
                            st.markdown("---")
                            st.subheader(translate_text("Transaction & Payment", current_lang))

                            if seller_upi_id:
                                st.success(translate_text(f"**Pay Seller via UPI (e.g., GPay, PhonePe, Paytm):** `{seller_upi_id}`", current_lang), icon="💳")
                                st.markdown(translate_text(f"You can use this UPI ID in your preferred UPI app to pay **₹{asking_price:,.0f}** to **{seller_full_name or seller_username}**.", current_lang))
                                
                                # Basic UPI Link (might not always open GPay directly but shows options)
                                upi_payment_string = f"upi://pay?pa={seller_upi_id}&pn={seller_full_name or seller_username}&am={asking_price:.2f}&cu=INR&tn=Payment for Cattle ID {listing_id}"
                                st.link_button(translate_text("Generate UPI Payment Link (Experimental)", current_lang), upi_payment_string, help=translate_text("Click to try opening UPI app. May not work on all devices/browsers.", current_lang))
                                
                                st.markdown(translate_text("""
                                **Important Steps After Payment:**
                                1. Complete the payment using your UPI app.
                                2. **Share the transaction ID/screenshot with the seller** as proof of payment.
                                3. Coordinate with the seller for cattle pickup/delivery.
                                4. Once the transaction is mutually confirmed, the seller should update the listing status.
                                """, current_lang))
                            else:
                                st.info(translate_text("Seller has not provided a UPI ID for direct payment. Please coordinate payment directly with the seller using their other contact details (if shared).", current_lang))

                            st.markdown(translate_text("**Note:** This platform currently does not process payments directly. All financial transactions are to be handled between the buyer and seller. Please ensure to verify all details before making payments.", current_lang))
                            
                            if st.button(translate_text(f"🤝 Mark as 'Interested / Payment Discussed'", current_lang, key="auto_btn_14"), key=f"mark_interested_{listing_id}"):
                                st.success(translate_text("Your interest has been noted. Please contact the seller to finalize the purchase and payment.", current_lang))
                                # This is where you would ideally log this interest in your database.

                            is_already_saved_to_wishlist = False
                            if st.session_state.logged_in:
                                try:
                                    cursor.execute("""SELECT saved_id FROM user_saved_listings
                                                    WHERE user_id = ? AND listing_type = 'cattle' AND original_listing_id = ?""",
                                                   (st.session_state.user_id, listing_id)) 
                                    if cursor.fetchone():
                                        is_already_saved_to_wishlist = True
                                except sqlite3.Error as e_check_save:
                                    logger.error(f"Error checking wishlist for cattle {listing_id}: {e_check_save}")

                            if st.session_state.logged_in:
                                if is_already_saved_to_wishlist:
                                    st.button(translate_text("❤️ Tracking This Item", current_lang, key="auto_btn_15"), key=f"wishlist_saved_cattle_{listing_id}", disabled=True, type="primary")
                                else:
                                    if st.button(translate_text("➕ Mark as Interested & Track", current_lang, key="auto_btn_16"), key=f"wishlist_save_cattle_{listing_id}"):
                                        try:
                                            cursor.execute("""INSERT INTO user_saved_listings
                                                            (user_id, listing_type, original_listing_id)
                                                            VALUES (?, 'cattle', ?)""",
                                                           (st.session_state.user_id, listing_id))
                                            conn.commit()
                                            st.toast(translate_text(f" '{animal_name or 'Cattle'}' is now being tracked in 'Saved Alerts / Wishlist'!", current_lang), icon="❤️")
                                            st.rerun()
                                        except sqlite3.IntegrityError:
                                            st.toast(translate_text("This item is already being tracked.", current_lang), icon="✅")
                                        except sqlite3.Error as e_save_wish:
                                            st.error(translate_text(f"Error saving to tracking list: {e_save_wish}", current_lang))
            else:
                st.info(translate_text("No cattle listings match your criteria, or no cattle are currently for sale.", current_lang))
            
        except sqlite3.Error as e:
            st.error(translate_text(f"Could not fetch cattle listings: {e}", current_lang))
            logger.error(f"DB error Browse cattle: {e}")

# --- In your main PAGE CONTENT ROUTING section (elif st.session_state.current_page == ...) ---

# --- In your main PAGE CONTENT ROUTING section (elif st.session_state.current_page == ...) ---

elif st.session_state.current_page == "Farm Products":  # Renamed from "Find a Farmer"
    ts.title("🛍️ " + translate_text("Farm Products Marketplace", current_lang))
    ts.markdown(translate_text("Discover fresh, local products directly from farmers. Farmers can list their offerings here.", current_lang))
    st.markdown("---")

    conn = get_connection()
    if not conn:
        st.error(translate_text("Database connection failed.", current_lang))
    else:
        cursor = conn.cursor()

        # --- Section for Logged-in Farmers to Edit Their Product Offerings ---
        if st.session_state.logged_in and st.session_state.role == 'farmer':
            ts.subheader(translate_text(f"📢 Manage Your Farm's Product Listings, {st.session_state.username}", current_lang))
            with st.expander(translate_text("✏️ Edit My Product Offerings & Location", current_lang), expanded=False):
                try:
                    cursor.execute("""SELECT full_name, email, phone_number, address, region,
                                            latitude, longitude, sells_products, product_categories, share_contact_info, upi_id
                                       FROM users WHERE user_id = ?""", (st.session_state.user_id,))
                    farmer_profile = cursor.fetchone()
                except sqlite3.Error as e_prof:
                    st.error(translate_text(f"Could not load your profile: {e_prof}", current_lang))
                    farmer_profile = None

                if farmer_profile:
                    (prof_fname, prof_email, prof_phone, prof_address, prof_region,
                     prof_lat, prof_lon, prof_sells, prof_prod_cat, prof_share, prof_upi_id) = farmer_profile

                    with st.form(translate_text("edit_farmer_products_form_fp", current_lang)): # Unique key for the form
                        ts.markdown(translate_text("Indicate if you sell farm products and specify categories.", current_lang))
                        new_sells_products = st.checkbox(translate_text("I sell farm products", current_lang), value=bool(prof_sells), key="fp_sells_check_edit")
                        new_product_categories = st.text_area(translate_text("Product Categories (comma-separated)", current_lang),
                                                                value=prof_prod_cat or "",
                                                                placeholder=translate_text("e.g., Fresh Milk, A2 Ghee, Organic Manure", current_lang),
                                                                key="fp_prod_cat_input_edit",
                                                                height=100,
                                                                disabled=not new_sells_products)
                        st.markdown("---")
                        ts.markdown(translate_text("**Your Farm Location (for map display):**", current_lang))
                        col_loc1_fp, col_loc2_fp = st.columns(2)
                        new_address_fp = col_loc1_fp.text_input(translate_text("Farm Address", current_lang), value=prof_address or "", key="fp_addr_edit")
                        new_region_fp = col_loc2_fp.text_input(translate_text("District, State", current_lang), value=prof_region or "", key="fp_region_state_edit")
                        col_geo1_fp, col_geo2_fp = st.columns(2)
                        new_latitude_fp = col_geo1_fp.number_input(translate_text("Latitude", current_lang), value=float(prof_lat) if prof_lat is not None else None, format="%.6f", key="fp_lat_edit", help=translate_text("Right-click on Google Maps to get coordinates", current_lang))
                        new_longitude_fp = col_geo2_fp.number_input(translate_text("Longitude", current_lang), value=float(prof_lon) if prof_lon is not None else None, format="%.6f", key="fp_lon_edit")
                        st.markdown("---")
                        new_share_contact_fp = st.checkbox(translate_text("Share my contact details on listings?", current_lang), value=bool(prof_share), key="fp_share_contact_edit")
                        
                        # Add UPI ID input
                        new_upi_id_fp = st.text_input(translate_text("Your UPI ID (for direct payments)", current_lang), value=prof_upi_id or "", help=translate_text("e.g., yourname@bankname or yourphonenumber@upi", current_lang))


                        if st.form_submit_button(translate_text("Update My Info", current_lang)):
                            try:
                                cursor.execute("""UPDATE users SET
                                                    address = ?, region = ?, latitude = ?, longitude = ?,
                                                    sells_products = ?, product_categories = ?, share_contact_info = ?, upi_id = ?
                                                WHERE user_id = ?""",
                                               (new_address_fp or None, new_region_fp or None, new_latitude_fp, new_longitude_fp,
                                                1 if new_sells_products else 0, new_product_categories.strip() if new_sells_products else None,
                                                1 if new_share_contact_fp else 0, new_upi_id_fp.strip() if new_upi_id_fp.strip() else None,
                                                st.session_state.user_id))
                                conn.commit()
                                st.success(translate_text("Your product and location info updated!", current_lang)); st.rerun()
                            except sqlite3.Error as e_upd: st.error(translate_text(f"DB error: {e_upd}", current_lang))
            st.markdown("---") # End of farmer's edit section


        # --- Public Browse Section (for all users) ---
        ts.subheader(translate_text("Browse Products from Local Farmers", current_lang))

        # Filters
        filter_col_prod1, filter_col_prod2 = st.columns(2)
        search_product_cat_fp = filter_col_prod1.text_input(translate_text("Search by Product Category (e.g., Milk, Ghee)", current_lang), key="fp_search_prod_cat_main")
        search_farmer_loc_fp = filter_col_prod2.text_input(translate_text("Search by Farmer Location (Region/State/District)", current_lang), key="fp_search_farm_loc_main")

        query_farmers_selling_fp = """
            SELECT user_id, username, full_name, region, address, latitude, longitude,
                   product_categories, phone_number, email, share_contact_info, upi_id
            FROM users
            WHERE role = 'farmer' AND sells_products = 1
        """
        params_farmers_selling_fp = []

        if search_product_cat_fp:
            query_farmers_selling_fp += " AND LOWER(product_categories) LIKE ?"
            params_farmers_selling_fp.append(f"%{search_product_cat_fp.lower()}%")
        if search_farmer_loc_fp:
            query_farmers_selling_fp += " AND (LOWER(region) LIKE ? OR LOWER(address) LIKE ?)" # Search in both region and address
            params_farmers_selling_fp.extend([f"%{search_farmer_loc_fp.lower()}%", f"%{search_farmer_loc_fp.lower()}%"])
        
        query_farmers_selling_fp += " ORDER BY region, COALESCE(full_name, username)"

        try:
            cursor.execute(query_farmers_selling_fp, params_farmers_selling_fp)
            farmers_with_products_fp = cursor.fetchall()

            if not farmers_with_products_fp:
                st.info(translate_text("No farmers found matching your criteria or listing products currently.", current_lang))
            else:
                map_data_list_fp = []
                displayable_farmers_list = [] # Farmers to show in the list section

                for farmer_row in farmers_with_products_fp:
                    # farmer_row format: (user_id, username, full_name, region, address, latitude, longitude, product_categories, phone_number, email, share_contact_info, upi_id)
                    displayable_farmers_list.append(farmer_row) # Add all for list view initially
                    
                    lat_val = farmer_row[5] # latitude
                    lon_val = farmer_row[6] # longitude
                    if lat_val is not None and lon_val is not None:
                        try:
                            # Add to map_data_list if lat/lon are valid
                            map_data_list_fp.append({'lat': float(lat_val), 'lon': float(lon_val), 'size': 20})
                        except (TypeError, ValueError):
                            logger.warning(f"Farmer {farmer_row[1]} has invalid lat/lon for map: {lat_val}, {lon_val}")
                
                # --- MAP DISPLAY (RESTORED) ---
                if map_data_list_fp:
                    df_map_fp = pd.DataFrame(map_data_list_fp)
                    ts.markdown(translate_text("##### Farmer Locations on Map (Approximate):", current_lang))
                    st.map(df_map_fp, zoom=4) # Adjust default zoom level as needed
                    st.markdown("---")
                elif farmers_with_products_fp: # If there are farmers but none had map data
                    st.info(translate_text("No farmers with valid location data to display on the map for current filters.", current_lang))
                # --- END MAP DISPLAY ---
                
                ts.markdown(translate_text(f"##### Found {len(displayable_farmers_list)} Farmer(s) Selling Products:", current_lang))
                for farmer_detail_tuple in displayable_farmers_list:
                    (user_id_d, username_d, full_name_d, region_d, address_d,
                     _lat_d, _lon_d, # We don't need to display lat/lon in the card directly
                     product_categories_d, phone_d, email_d, share_contact_d, upi_id_d) = farmer_detail_tuple

                    with st.container(border=True):
                        ts.subheader(f"{full_name_d or username_d}")
                        if region_d: ts.write(translate_text(f"**Region:** {region_d}", current_lang))
                        
                        if product_categories_d:
                            ts.markdown(translate_text(f"**Selling:**", current_lang))
                            categories_list = [cat.strip() for cat in product_categories_d.split(',')]
                            # Simple list display for categories
                            for cat_item in categories_list:
                                ts.markdown(f"- {cat_item}")
                        else:
                            ts.write(translate_text("Products: Not specified by the farmer.", current_lang))

                        # Expander for more details and contact
                        with st.expander(translate_text("View More Details & Contact", current_lang)):
                            if address_d: ts.write(translate_text(f"**Farm Address (Approx.):** {address_d}", current_lang))
                            if product_categories_d: ts.write(translate_text(f"**Full Product List:** {product_categories_d.replace(',',', ')}", current_lang)) # Re-list if needed

                            if share_contact_d == 1:
                                ts.markdown(translate_text("**Contact Information:**", current_lang))
                                if phone_d: ts.write(translate_text(f"📞 Phone: `{phone_d}`", current_lang))
                                if email_d: ts.write(translate_text(f"📧 Email: [{email_d}](mailto:{email_d}?subject=Inquiry about Farm Products from {full_name_d or username_d})", current_lang))
                                if not phone_d and not email_d: st.caption(translate_text("Contact details not fully provided by seller.", current_lang))
                            else:
                                st.caption(translate_text("Seller has chosen not to share direct contact details publicly.", current_lang))

                            # Add Payment Facilitation Section for Farm Products
                            ts.markdown("---")
                            ts.subheader(translate_text("Payment Information", current_lang))
                            if upi_id_d:
                                ts.success(translate_text(f"**Pay Farmer via UPI (e.g., GPay, PhonePe):** `{upi_id_d}`", current_lang), icon="💳")
                                ts.markdown(translate_text(f"You can use this UPI ID to pay **{full_name_d or username_d}** directly for their products.", current_lang))
                                # You can optionally generate a generic UPI payment link here if there's no fixed price.
                                # For farm products, prices are dynamic, so a direct link might not always make sense.
                                # Example if you wanted a generic link:
                                # upi_generic_link = f"upi://pay?pa={upi_id_d}&pn={full_name_d or username_d}&cu=INR"
                                # ts.link_button(translate_text("Initiate UPI Payment", current_lang), upi_generic_link, help=translate_text("Opens UPI app for direct payment.", current_lang))
                                ts.markdown(translate_text("""**Important Steps After Payment:**
1. Coordinate exact amount and terms with the farmer.
2. Complete payment using your UPI app.
3. **Share transaction ID/screenshot with farmer** as proof.
4. Coordinate for product pickup/delivery.
""", current_lang))
                            elif share_contact_d == 1 and (email_d or phone_d):
                                st.info(translate_text("Farmer has not provided a UPI ID. Please use contact details above to discuss payment and product terms.", current_lang))
                            else:
                                st.info(translate_text("Farmer has not provided UPI ID or public contact details. Please try other listings.", current_lang))

                            ts.markdown(translate_text("**Note:** This platform does not process payments directly.", current_lang))

                        st.markdown("---") # Separator for each farmer
        except sqlite3.Error as e:
            st.error(translate_text(f"Database error fetching farmer product data: {e}", current_lang))
            logger.error(f"DB error Farm Products Marketplace: {e}")
        # No conn.close() here due to @st.cache_resource

elif st.session_state.current_page == "Vet Locator" and st.session_state.logged_in and st.session_state.role == "farmer":
    ts.title("👩‍⚕️ Veterinarian Locator (Map-Based)")
    ts.markdown("Find veterinarians near you. Zoom in on the map for details.")
    st.caption(translate_text("Note: Veterinarian listings are based on available data. Always verify credentials and availability.",current_lang))
    st.markdown("---")

    conn = get_connection()
    if not conn:
        st.error(translate_text("Database connection failed.",current_lang))
    else:
        cursor = conn.cursor()

        # Filters
        filter_col1, filter_col2 = st.columns(2)
        # For simplicity, using text input for city/state. Could be dropdowns populated from DB.
        search_city_vet = filter_col1.text_input("Search by City:", key="vet_search_city")
        search_specialization_vet = filter_col2.text_input("Search by Specialization (e.g., Bovine):", key="vet_search_spec")

        query_vets = """
            SELECT vet_id, name, clinic_name, specialization, address, city, state,
                   phone_number, email, latitude, longitude, services_offered, operating_hours
            FROM veterinarians
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        """ # Only select vets with coordinates for map
        params_vets = []

        if search_city_vet:
            query_vets += " AND LOWER(city) LIKE ?"
            params_vets.append(f"%{search_city_vet.lower()}%")
        if search_specialization_vet:
            query_vets += " AND LOWER(specialization) LIKE ?"
            params_vets.append(f"%{search_specialization_vet.lower()}%")
        query_vets += " ORDER BY state, city, name"


        try:
            cursor.execute(query_vets, params_vets)
            vets_data = cursor.fetchall()

            if not vets_data:
                st.info(translate_text("No veterinarians found matching your criteria or no vet data available.",current_lang))
            else:
                map_data_vets_list = []
                for vet in vets_data:
                    lat_v = vet[9] # latitude is at index 9
                    lon_v = vet[10] # longitude is at index 10
                    if lat_v and lon_v: # Ensure not None
                        map_data_vets_list.append({'lat': float(lat_v), 'lon': float(lon_v)})

                if map_data_vets_list:
                    df_map_vets = pd.DataFrame(map_data_vets_list)
                    ts.subheader("Veterinarian Locations:")
                    st.map(df_map_vets, zoom=5) # Adjust default zoom
                    st.markdown("---")
                else:
                    st.info(translate_text("No veterinarians with valid location data found for the current filters.",current_lang))


                ts.subheader("Veterinarian Listings:")
                for vet in vets_data:
                    (vet_id, name, clinic_name, specialization, address, city, state,
                     phone, email, lat_v, lon_v, services, hours) = vet # Unpack all

                    with st.container(border=True):
                        ts.subheader(f"Dr. {name} {f'- {clinic_name}' if clinic_name else ''}")
                        if specialization: ts.write(f"**Specialization:** {specialization}")
                        ts.write(f"**Address:** {address}, {city}, {state}")
                        if services: ts.write(f"**Services:** {services.replace(',', ', ')}")
                        if hours: ts.write(f"**Hours:** {hours}")

                        ts.markdown("**Contact:**")
                        if phone: ts.write(f"📞 Phone: `{phone}`")
                        if email: ts.write(f"📧 Email: [{email}](mailto:{email})")
                        st.markdown("---")
        except sqlite3.Error as e:
            st.error(translate_text(f"Database error fetching veterinarian data: {e}",current_lang))
            logger.error(f"DB error Vet Locator: {e}")
        # No conn.close() due to @st.cache_resource

elif st.session_state.current_page == "Browse Machinery":  # Assuming access control is done
    ts.title("🚜 " + translate_text("Browse Farm Machinery for Sale/Rent", current_lang))
    ts.markdown(translate_text("Find machinery listed by other users.", current_lang))
    ts.markdown("---")

    conn = get_connection()
    if not conn:
        ts.error(translate_text("Database connection failed.", current_lang))
        # Consider st.stop() here if you don't want to render the rest of the page
    else:
        cursor = conn.cursor()

        # --- Filters for Machinery ---
        bm_c1_bm, bm_c2_bm, bm_c3_bm = st.columns(3)
        search_mach_name_bm_filter = bm_c1_bm.text_input(translate_text("Search Name/Type/Brand", current_lang), key="bm_search_name_input_filter_v2_corrected") # Corrected key
        search_mach_loc_bm_filter = bm_c2_bm.text_input(translate_text("Location", current_lang), key="bm_search_loc_input_filter_v2_corrected")      # Corrected key
        mach_type_filter_options_bm = [translate_text("All Types", current_lang)] + [
            translate_text("Tillage", current_lang),
            translate_text("Sowing/Planting", current_lang),
            translate_text("Harvesting", current_lang),
            translate_text("Post-Harvest", current_lang),
            translate_text("Dairy Equipment", current_lang),
            translate_text("Irrigation", current_lang),
            translate_text("Transport", current_lang),
            translate_text("Spares/Parts", current_lang),
            translate_text("Other", current_lang)
        ]
        search_mach_type_bm_filter = bm_c3_bm.selectbox(translate_text("Filter Type", current_lang), options=mach_type_filter_options_bm, key="bm_search_type_select_filter_v2_corrected") # Corrected key

        # --- Save Alert Snippet Begins Here ---
        current_mach_filters = {}
        if search_mach_name_bm_filter:
            current_mach_filters['type_name_brand'] = search_mach_name_bm_filter # Changed key to be more descriptive
        if search_mach_loc_bm_filter:
            current_mach_filters['location'] = search_mach_loc_bm_filter
        if search_mach_type_bm_filter != translate_text("All Types", current_lang): # Compare with translated string
            current_mach_filters['machinery_type'] = search_mach_type_bm_filter # Changed key to be more descriptive

        if current_mach_filters and st.session_state.logged_in and st.session_state.role == 'buyer': # Only show save alert for logged-in buyers with active filters
            alert_name_mach = ts.text_input(translate_text("Save current search as Alert Name:", current_lang), key="save_alert_mach_name_input", placeholder=translate_text("e.g., Used Tractors Punjab", current_lang)) # Corrected key
            if ts.button(translate_text("💾 Save Alert for Machinery Search", current_lang), key="save_alert_mach_button"): # Corrected key
                if alert_name_mach.strip():
                    # Removed redundant get_connection() here, use existing 'conn' and 'cursor'
                    try:
                        criteria_json_str_m = json.dumps(current_mach_filters)
                        cursor.execute("""
                            INSERT INTO saved_alerts (user_id, alert_name, alert_type, criteria_json, created_at, last_checked_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            st.session_state.user_id,
                            alert_name_mach.strip(),
                            "machinery",
                            criteria_json_str_m,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # created_at
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # initial last_checked_at
                        ))
                        conn.commit() # Commit the transaction
                        ts.success(translate_text(f"Alert '{alert_name_mach}' saved! Check 'Saved Alerts' page.", current_lang))
                        # Consider st.rerun() if you want the input to clear or list of alerts to refresh immediately elsewhere
                    except sqlite3.Error as e_am:
                        ts.error(translate_text(f"Error saving machinery alert: {e_am}", current_lang))
                        logger.error(f"Error saving machinery alert for user {st.session_state.user_id}: {e_am}")
                else:
                    ts.warning(translate_text("Please provide a name for your alert to save it.", current_lang))
        ts.markdown("---")
        # --- Save Alert Snippet Ends Here ---

        query_mach_bm = """
            SELECT
                ml.machinery_id,
                u.username as seller_username,
                seller_profile.full_name as seller_full_name,
                seller_profile.region as seller_region,
                seller_profile.email as seller_email,
                seller_profile.phone_number as seller_phone,
                seller_profile.share_contact_info,
                seller_profile.upi_id as seller_upi, -- Added for payment
                ml.name, ml.type, ml.brand, ml.model,
                ml.year_of_manufacture, ml.condition, ml.asking_price,
                ml.description, ml.location as listing_location, ml.listing_date,
                ml.image_url_1, ml.image_url_2, ml.for_rent, ml.rental_price_day
            FROM machinery_listings ml
            JOIN users u ON ml.user_id = u.user_id
            JOIN users seller_profile ON ml.user_id = seller_profile.user_id
            WHERE ml.status = 'Available'
        """
        params_mach_bm = []
        if search_mach_name_bm_filter:
            query_mach_bm += " AND (LOWER(ml.name) LIKE ? OR LOWER(ml.type) LIKE ? OR LOWER(ml.brand) LIKE ? OR LOWER(ml.model) LIKE ?)"
            term_bm = f"%{search_mach_name_bm_filter.lower()}%"
            params_mach_bm.extend([term_bm, term_bm, term_bm, term_bm])
        if search_mach_loc_bm_filter:
            query_mach_bm += " AND LOWER(ml.location) LIKE ?"
            params_mach_bm.append(f"%{search_mach_loc_bm_filter.lower()}%")
        if search_mach_type_bm_filter != translate_text("All Types", current_lang): # Compare with translated string
            query_mach_bm += " AND ml.type = ?" # Assuming your machinery table has 'type' not 'condition' for this filter
            params_mach_bm.append(search_mach_type_bm_filter) # Use the translated type
        query_mach_bm += " ORDER BY ml.listing_date DESC"

        try:
            cursor.execute(query_mach_bm, params_mach_bm)
            mach_listings_data = cursor.fetchall() # Renamed from your snippet
            if mach_listings_data:
                ts.subheader(translate_text(f"Found {len(mach_listings_data)} machinery listings:", current_lang))
                for mach_listing_tuple in mach_listings_data: # Use the new variable name
                    # Ensure unpacking matches the SELECT statement order and count (now 21 columns with seller_upi)
                    (machinery_id, seller_username, seller_full_name, seller_region, seller_email, 
                     seller_phone, seller_share_contact, seller_upi,
                     name, type_mach, brand, model, yom, condition, asking_price,
                     description, listing_location, listing_date_str,
                     image_url_1, image_url_2, for_rent, rental_price_day) = mach_listing_tuple
                    
                    is_already_saved_machinery = False
                    if st.session_state.logged_in:
                        try:
                            cursor.execute("""SELECT saved_id FROM user_saved_listings
                                            WHERE user_id = ? AND listing_type = 'machinery' AND original_listing_id = ?""",
                                           (st.session_state.user_id, machinery_id))
                            if cursor.fetchone():
                                is_already_saved_machinery = True
                        except sqlite3.Error as e_check_save_m:
                            logger.error(f"Error checking saved machinery {machinery_id}: {e_check_save_m}")

                    with st.container(border=True):
                        col_summary_img_mach, col_summary_info_mach = st.columns([1, 3])
                        with col_summary_img_mach:
                            display_uploaded_image(image_url_1, caption=name, use_container_width=True) # Assuming display_uploaded_image is defined
                        with col_summary_info_mach:
                            ts.subheader(translate_text(f"{name} ({brand or ''} {model or ''})", current_lang))
                            ts.markdown(translate_text(f"**Type:** {type_mach} | **Condition:** {condition} | **YoM:** {yom or 'N/A'}", current_lang))
                            ts.markdown(translate_text(f"**Location:** {listing_location} ", current_lang))

                            price_display_mach_str = ""
                            if asking_price and asking_price > 0: price_display_mach_str += translate_text(f"Sale: ₹{asking_price:,.0f}", current_lang)
                            if for_rent == 1: # Check if for_rent is 1 (True)
                                rent_str_mach = translate_text(f"Rent: ₹{rental_price_day:,.0f}/day", current_lang) if rental_price_day and rental_price_day > 0 else translate_text("For Rent", current_lang)
                                price_display_mach_str = f"{price_display_mach_str} | {rent_str_mach}" if price_display_mach_str else rent_str_mach
                            ts.markdown(translate_text(f"**{price_display_mach_str or 'Price on Request'}**", current_lang))
                            ts.caption(translate_text(f"Listed by: {seller_username} on {datetime.strptime(listing_date_str, '%Y-%m-%d %H:%M:%S').strftime('%d %b %Y')}", current_lang))

                            is_this_machinery_expanded = (st.session_state.expanded_machinery_listing_id == machinery_id)
                            button_label_mach = translate_text("➖ Hide Details", current_lang) if is_this_machinery_expanded else translate_text("⚙️ View Full Details & Contact", current_lang)

                            if st.button(button_label_mach, key=f"view_mach_expander_{machinery_id}_corrected"): # Added corrected key
                                if is_this_machinery_expanded:
                                    st.session_state.expanded_machinery_listing_id = None
                                else:
                                    st.session_state.expanded_machinery_listing_id = machinery_id
                                    st.session_state.expanded_cattle_listing_id = None # Close other type if open
                                st.rerun()

                        if is_this_machinery_expanded: # This block is now correctly OUTSIDE col_summary_info_mach but INSIDE the main listing container
                            ts.markdown("---") # Visual separator for the expander content
                            ts.markdown(translate_text(f"##### Full Details for: {name}", current_lang))
                            ts.markdown(translate_text(f"**Listing ID:** {machinery_id}", current_lang))
                            ts.markdown(translate_text(f"**Full Description:**\n{description or 'No further description.'}", current_lang))
                            if model: ts.markdown(translate_text(f"**Model:** {model}", current_lang))
                            if yom: ts.markdown(translate_text(f"**Year of Manufacture:** {yom}", current_lang))
                            if image_url_2:
                                ts.markdown(translate_text("**Additional Image:**", current_lang))
                                display_uploaded_image(image_url_2, caption=translate_text("Additional Image", current_lang), use_container_width=True) # Assuming helper function

                            ts.markdown("---")
                            
                            ts.subheader(translate_text(f"Seller Information & Payment", current_lang)) # Moved outside the if logged_in for wishlist
                            ts.markdown(translate_text(f"**Seller:** {seller_full_name or seller_username} (Region: {seller_region or 'N/A'})", current_lang))
                            if seller_share_contact == 1:
                                ts.markdown(translate_text("**Contact Details:**", current_lang))
                                if seller_email: ts.markdown(translate_text(f"📧 Email: [{seller_email}](mailto:{seller_email}?subject=Inquiry about Machinery: {name} - ID {machinery_id})", current_lang))
                                if seller_phone: ts.markdown(translate_text(f"📞 Phone: `{seller_phone}`", current_lang))
                                ts.caption(translate_text("Please be respectful when contacting sellers.", current_lang))
                            else:
                                ts.info(translate_text("Seller has chosen not to share direct contact details publicly.", current_lang))

                            # --- Payment Facilitation Section ---
                            if seller_upi: # seller_upi was fetched from seller_profile.upi_id
                                ts.success(translate_text(f"**Pay Seller via UPI (e.g., GPay, PhonePe):** `{seller_upi}`", current_lang), icon="💳")
                                if asking_price and asking_price > 0:
                                    ts.markdown(translate_text(f"You can use this UPI ID to pay **₹{asking_price:,.0f}** to **{seller_full_name or seller_username}**.", current_lang))
                                    upi_payment_str_mach = f"upi://pay?pa={seller_upi}&pn={seller_full_name or seller_username}&am={asking_price:.2f}&cu=INR&tn=Payment for Machinery ID {machinery_id}"
                                    ts.link_button(translate_text("Generate UPI Payment Link (Experimental)", current_lang), upi_payment_str_mach, help=translate_text("Click to try opening UPI app.", current_lang))
                                elif for_rent == 1 and rental_price_day and rental_price_day > 0:
                                    ts.markdown(translate_text(f"For rental, coordinate amount & terms using UPI ID: `{seller_upi}`.", current_lang))
                                else:
                                    ts.markdown(translate_text(f"Coordinate amount & terms directly using UPI ID: `{seller_upi}`.", current_lang))
                                ts.markdown(translate_text("""**Important Steps After Payment:**
1. Complete payment using your UPI app.
2. **Share transaction ID/screenshot with seller** as proof.
3. Coordinate for machinery pickup/delivery.
""", current_lang))
                            elif seller_share_contact == 1 and (seller_email or seller_phone):
                                ts.info(translate_text("Seller has not provided a UPI ID. Use contact details above to discuss payment.", current_lang))
                            else:
                                ts.info(translate_text("Seller has not provided UPI ID or public contact details.", current_lang))
                            
                            ts.markdown(translate_text("**Note:** This platform does not process payments directly.", current_lang))
                        is_already_saved_mach_wishlist = False
                        if st.session_state.logged_in:
                            try:
                                cursor.execute("""SELECT saved_id FROM user_saved_listings
                                                 WHERE user_id = ? AND listing_type = 'machinery' AND original_listing_id = ?""",
                                               (st.session_state.user_id, machinery_id)) # machinery_id from machinery_listings
                                if cursor.fetchone():
                                    is_already_saved_mach_wishlist = True
                            except sqlite3.Error as e_check_save_m_wish:
                                logger.error(f"Error checking wishlist for machinery {machinery_id}: {e_check_save_m_wish}")
                        
                        if st.session_state.logged_in:
                            if is_already_saved_mach_wishlist:
                                st.button(translate_text("❤️ Tracking This Item", current_lang, key="auto_btn_18"), key=f"wishlist_saved_mach_{machinery_id}", disabled=True, type="primary")
                            else:
                                if st.button(translate_text("➕ Mark as Interested & Track", current_lang, key="auto_btn_19"), key=f"wishlist_save_mach_{machinery_id}"):
                                    try:
                                        cursor.execute("""INSERT INTO user_saved_listings
                                                             (user_id, listing_type, original_listing_id)
                                                             VALUES (?, 'machinery', ?)""",
                                                           (st.session_state.user_id, machinery_id))
                                        conn.commit()
                                        st.toast(translate_text(f" '{name}' is now being tracked in 'Saved Alerts / Wishlist'!", current_lang), icon="❤️") # name is machinery name
                                        st.rerun()
                                    except sqlite3.IntegrityError:
                                        ts.toast(translate_text("This item is already being tracked.", current_lang), icon="✅")
                                    except sqlite3.Error as e_save_m_wish:
                                        ts.error(translate_text(f"Error saving to tracking list: {e_save_m_wish}", current_lang))
                        ts.markdown("---") # Separator for each main listing item
            else:
                ts.info(translate_text("No machinery listings match your criteria.", current_lang))
        except sqlite3.Error as e:
            ts.error(translate_text(f"Could not fetch machinery listings: {e}", current_lang))
            logger.error(f"DB error Browse machinery: {e}")


# --- MODIFIED: Sell Machinery Page (Farmer/Buyer) ---
elif selected_page == "Sell Machinery" and st.session_state.logged_in:
    ts.title("🛠️ " + translate_text("List Your Farm Machinery for Sale/Rent", current_lang))
    ts.markdown(translate_text("Offer your used or surplus machinery to other users.", current_lang))
    ts.markdown("---")
    conn = get_connection()
    if not conn: ts.error(translate_text("Database connection failed.", current_lang))
    else:
        cursor = conn.cursor()
        # Change 'ts.form' back to 'st.form'
        with st.form("sell_machinery_form", clear_on_submit=True):
            ts.subheader(translate_text("Machinery Details", current_lang))
            m_c1_sm, m_c2_sm = st.columns(2)
            mach_name_sm = m_c1_sm.text_input(translate_text("Machinery Name/Title*", current_lang), help=translate_text("e.g., Tractor - Mahindra 265 DI, Used Plough", current_lang))
            mach_type_options_sm = [
                translate_text("Tillage", current_lang),
                translate_text("Sowing/Planting", current_lang),
                translate_text("Harvesting", current_lang),
                translate_text("Post-Harvest", current_lang),
                translate_text("Dairy Equipment", current_lang),
                translate_text("Irrigation", current_lang),
                translate_text("Transport", current_lang),
                translate_text("Spares/Parts", current_lang),
                translate_text("Other", current_lang)
            ]
            mach_type_sm = m_c2_sm.selectbox(translate_text("Type/Category*", current_lang), options=mach_type_options_sm)

            m_c3_sm, m_c4_sm =st.columns(2)
            mach_brand_sm = m_c3_sm.text_input(translate_text("Brand (if any)", current_lang))
            mach_model_sm = m_c4_sm.text_input(translate_text("Model (if any)", current_lang))
            
            m_c5_sm, m_c6_sm = st.columns(2)
            current_year = datetime.now().year
            mach_yom_sm = m_c5_sm.number_input(translate_text("Year of Manufacture (approx)", current_lang), min_value=1950, max_value=current_year, value=current_year - 5, step=1)
            mach_condition_options_sm = [
                translate_text("New", current_lang),
                translate_text("Used - Excellent", current_lang),
                translate_text("Used - Good", current_lang),
                translate_text("Used - Fair", current_lang),
                translate_text("Needs Repair", current_lang),
                translate_text("For Spares", current_lang)
            ]
            mach_condition_sm = m_c6_sm.selectbox(translate_text("Condition*", current_lang), options=mach_condition_options_sm)
            
            m_c7_sm, m_c8_sm = st.columns([3,1])
            mach_price_sm = m_c7_sm.number_input(translate_text("Asking Price (₹)*", current_lang), min_value=0.0, step=100.0, help=translate_text("Enter 0 if only for rent or free.", current_lang))
            # Change 'ts.checkbox' to 'st.checkbox'
            mach_for_rent_sm = m_c8_sm.checkbox(translate_text("For Rent?", current_lang), value=False, key="mach_rent_check")
            mach_rental_price_day_sm = 0.0
            if mach_for_rent_sm:
                # Change 'ts.number_input' to 'st.number_input'
                mach_rental_price_day_sm = st.number_input(translate_text("Rental Price per Day (₹)", current_lang), min_value=0.0, step=50.0, key="mach_rent_price")


            # Change 'ts.text_area' to 'st.text_area'
            mach_desc_sm = st.text_area(translate_text("Description*", current_lang), height=150, help=translate_text("Include details like HP, capacity, hours used, any defects, reason for selling/renting.", current_lang))
            # Change 'ts.text_input' to 'st.text_input'
            mach_location_sm = st.text_input(translate_text("Your Location* (District, State)", current_lang), placeholder=translate_text("e.g., Nagpur, Maharashtra", current_lang))
            
            ts.markdown(translate_text("##### Upload Images (Optional, up to 2)", current_lang))
            img_col1_sm, img_col2_sm = st.columns(2)
            with img_col1_sm:
                # Change 'ts.file_uploader' to 'st.file_uploader'
                mach_image1_sm = st.file_uploader(translate_text("Image 1 (Main)", current_lang), type=['jpg', 'jpeg', 'png'], key="sell_mach_img1")
            with img_col2_sm:
                # Change 'ts.file_uploader' to 'st.file_uploader'
                mach_image2_sm = st.file_uploader(translate_text("Image 2 (Optional)", current_lang), type=['jpg', 'jpeg', 'png'], key="sell_mach_img2")


            # Change 'ts.form_submit_button' to 'st.form_submit_button'
            submit_mach_listing_sm = st.form_submit_button(translate_text("✅ List Machinery", current_lang))

            if submit_mach_listing_sm:
                if not mach_name_sm.strip() or not mach_type_sm or (mach_price_sm <= 0 and not mach_for_rent_sm) or not mach_desc_sm.strip() or not mach_location_sm.strip():
                    ts.error(translate_text("Please fill in Name, Type, Price (or select for rent), Description, and Location.", current_lang))
                elif mach_for_rent_sm and mach_rental_price_day_sm <=0 and mach_price_sm <=0:
                    ts.error(translate_text("If listing for rent, please provide a rental price (or an asking price if also for sale).", current_lang))
                else:
                    img1_path_sm = save_uploaded_image(mach_image1_sm, "machinery") if mach_image1_sm else None
                    img2_path_sm = save_uploaded_image(mach_image2_sm, "machinery") if mach_image2_sm else None
                    try:
                        cursor.execute("""
                            INSERT INTO machinery_listings (user_id, name, type, brand, model, year_of_manufacture,
                                                            condition, asking_price, description, location, status,
                                                            image_url_1, image_url_2, for_rent, rental_price_day)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Available', ?, ?, ?, ?)
                        """, (st.session_state.user_id, mach_name_sm, mach_type_sm, mach_brand_sm or None, mach_model_sm or None,
                                mach_yom_sm, mach_condition_sm, mach_price_sm if mach_price_sm > 0 else None, # Store None if price is 0
                                mach_desc_sm, mach_location_sm, img1_path_sm, img2_path_sm,
                                1 if mach_for_rent_sm else 0, mach_rental_price_day_sm if mach_for_rent_sm else None))
                        conn.commit()
                        ts.success(translate_text(f"Machinery '{mach_name_sm}' listed successfully!", current_lang))
                        logger.info(f"User {st.session_state.username} listed machinery: {mach_name_sm}")
                    except sqlite3.Error as e:
                        ts.error(translate_text(f"Database error: {e}", current_lang))
                        logger.error(f"DB error listing machinery for {st.session_state.username}: {e}")
            
        ts.markdown("---")
        ts.subheader(translate_text("Your Active Machinery Listings", current_lang))
        try:
            cursor.execute("""SELECT machinery_id, name, type, brand, model, asking_price, rental_price_day, for_rent, location, status, image_url_1
                            FROM machinery_listings WHERE user_id = ? AND status = 'Available'
                            ORDER BY listing_date DESC""", (st.session_state.user_id,))
            my_mach_listings_sm = cursor.fetchall()
            if my_mach_listings_sm:
                for mach_listing_sm_item in my_mach_listings_sm:
                    m_id, m_name, m_type, m_brand, m_model, m_price, m_rent_price, m_for_rent, m_loc, m_status, m_img1 = mach_listing_sm_item
                    # Change 'st.container' to 'ts.container' to be consistent with your 'ts' object
                    with ts.container(border=True):
                        disp_mc1, disp_mc2 = st.columns([1,3])
                        with disp_mc1:
                            display_uploaded_image(m_img1, caption=m_name, width=120)
                        with disp_mc2:
                            ts.subheader(translate_text(f"{m_name} ({m_brand or ''} {m_model or ''})", current_lang))
                            price_display = translate_text(f"Sale Price: ₹{m_price:,.0f}", current_lang) if m_price and m_price > 0 else translate_text("Not for direct sale", current_lang)
                            if m_for_rent == 1 and m_rent_price and m_rent_price > 0:
                                price_display += translate_text(f" | Rent: ₹{m_rent_price:,.0f}/day", current_lang)
                            elif m_for_rent == 1:
                                price_display += translate_text(" | Available for Rent (price on request)", current_lang)

                            ts.markdown(translate_text(f"**Type:** {m_type} | {price_display}", current_lang))
                            ts.caption(translate_text(f"Location: {m_loc} | Status: {m_status} | Listing ID: {m_id}", current_lang))
                            # TODO: Add edit/withdraw
                            # Change 'st.button' to 'ts.button'
                            if ts.button(translate_text("Withdraw Machinery Listing", current_lang), key=f"withdraw_mach_{m_id}", type="secondary"):
                                try:
                                    cursor.execute("UPDATE machinery_listings SET status = 'Withdrawn' WHERE machinery_id = ? AND user_id = ?", (m_id, st.session_state.user_id))
                                    conn.commit()
                                    ts.success(translate_text(f"Machinery Listing ID {m_id} withdrawn.", current_lang))
                                    st.rerun()
                                except sqlite3.Error as e_wd_m:
                                    ts.error(translate_text(f"Error withdrawing machinery: {e_wd_m}", current_lang))

                        ts.markdown("---")
            else:
                st.info(translate_text("You have no active machinery listings.", current_lang))
        except sqlite3.Error as e:
            ts.error(translate_text(f"Could not fetch your machinery listings: {e}", current_lang))

# --- In your main PAGE CONTENT ROUTING section ---

elif st.session_state.current_page == "Saved Alerts":
    # Access Control (ensure user is logged in and is a buyer, or adjust role as needed)
    if not st.session_state.logged_in or st.session_state.role != "buyer":
        st.warning("You must be logged in as a Buyer to view Saved Alerts & Tracked Items.")
        if st.button("Login", key="sa_login_btn_v6_final"): # Unique key
            st.session_state.current_page = "Login"
            st.rerun()
        st.stop() # Important to stop further rendering if not authorized

    ts.title("🔔 Saved Alerts & Tracked Items")
    ts.markdown("Manage your saved search criteria and view items you're actively tracking.")
    st.markdown("---")

    conn = get_connection()
    if not conn:
        st.error("Database connection failed. Please try again later.")
        st.stop() # Stop if DB connection fails

    cursor = conn.cursor()

    # --- Section 1: User-Created Search Alerts ---
    ts.subheader("📢 My Search Alerts")
    ts.caption("Create alerts to get notified about new listings that match your specific search criteria.")

    with st.expander("➕ Create New Search Alert", expanded=False):
        with st.form("create_alert_form_sa_v6_final", clear_on_submit=True):
            st.write("Define criteria for what you're looking for:")
            alert_name_sa_form = st.text_input("Alert Name*", help="e.g., 'Gir Cows Gujarat', 'Used Tractors MH'", key="sa_alert_name_form_v6_final")
            alert_type_sa_form = st.selectbox("Alert for*", ["Cattle", "Machinery"], key="sa_alert_type_form_v6_final")
            
            criteria_sa_form = {}

            if alert_type_sa_form == "Cattle":
                st.markdown("###### Cattle Criteria:")
                c_breed_sa_form = st.text_input("Breed (Optional)", key="sa_c_breed_form_v6_cattle_final", placeholder="e.g., Gir")
                c_loc_sa_form = st.text_input("Location (Optional)", key="sa_c_loc_form_v6_cattle_final", placeholder="e.g., Punjab")
                c_max_price_sa_form = st.number_input("Max Price (₹, Optional)", min_value=0, value=0, step=1000, key="sa_c_max_price_form_v6_cattle_final")
                
                if c_breed_sa_form.strip(): criteria_sa_form['breed'] = c_breed_sa_form.strip()
                if c_loc_sa_form.strip(): criteria_sa_form['location'] = c_loc_sa_form.strip()
                if c_max_price_sa_form > 0: criteria_sa_form['max_price'] = c_max_price_sa_form

            elif alert_type_sa_form == "Machinery":
                st.markdown("###### Machinery Criteria:")
                m_keyword_sa_form = st.text_input("Keyword (searches Name, Type, or Brand)", key="sa_m_keyword_form_v6_machinery_final", placeholder="e.g., Tractor, Plough, Mahindra")
                
                m_cond_options_form = ["Any Condition"] + ["New", "Used - Excellent", "Used - Good", "Used - Fair", "Needs Repair", "For Spares"]
                m_cond_sa_form = st.selectbox("Filter by Condition (Optional)", m_cond_options_form, key="sa_m_cond_form_v6_machinery_final")
                
                m_loc_sa_form = st.text_input("Filter by Location (Optional)", key="sa_m_loc_form_v6_machinery_final", placeholder="e.g., District or State")

                if m_keyword_sa_form.strip():
                    criteria_sa_form['keyword_machinery'] = m_keyword_sa_form.strip() # Key used in query
                if m_cond_sa_form != "Any Condition":
                    criteria_sa_form['condition'] = m_cond_sa_form # Key used in query
                if m_loc_sa_form.strip():
                    criteria_sa_form['location'] = m_loc_sa_form.strip() # Key used in query
            
            submit_alert_sa_form = st.form_submit_button("💾 Save Search Alert")

            if submit_alert_sa_form:
                if not alert_name_sa_form.strip():
                    st.error("Alert Name is required.")
                elif not criteria_sa_form:
                    st.error("Please specify at least one search criterion for the alert.")
                else:
                    try:
                        criteria_json_str_sa_form = json.dumps(criteria_sa_form)
                        current_ts_sa_form = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cursor.execute("""
                            INSERT INTO saved_alerts (user_id, alert_name, alert_type, criteria_json, created_at, last_checked_at, is_active)
                            VALUES (?, ?, ?, ?, ?, ?, 1)
                        """, (st.session_state.user_id, alert_name_sa_form.strip(), alert_type_sa_form.lower(),
                              criteria_json_str_sa_form, current_ts_sa_form, current_ts_sa_form))
                        conn.commit()
                        st.success(f"Search Alert '{alert_name_sa_form}' saved successfully!")
                        st.rerun()
                    except sqlite3.Error as e_save_alert:
                        st.error(f"Database error saving alert: {e_save_alert}")
                        logger.error(f"Error saving search alert for user {st.session_state.user_id}: {e_save_alert}", exc_info=True)
                    except Exception as e_gen_save_alert:
                        st.error(f"An unexpected error occurred while saving alert: {e_gen_save_alert}")
                        logger.error(f"Generic error saving search alert for user {st.session_state.user_id}: {e_gen_save_alert}", exc_info=True)
    
    st.markdown("---")
    ts.markdown("##### Your Active Search Alerts & New Matches:")
    try:
        cursor.execute("""
            SELECT alert_id, alert_name, alert_type, criteria_json, created_at, last_checked_at
            FROM saved_alerts WHERE user_id = ? AND is_active = 1 ORDER BY created_at DESC
        """, (st.session_state.user_id,))
        active_search_alerts = cursor.fetchall()

        if not active_search_alerts:
            st.info("You have no active search alerts. Create one above to get notified about new listings matching your criteria!")
        else:
            for alert_tuple_search in active_search_alerts:
                alert_id_search, name_search, type_search, crit_json_search, created_search, last_checked_search = alert_tuple_search
                criteria_search_display = json.loads(crit_json_search)
                session_matches_key_search = f"alert_matches_for_{alert_id_search}" # Unique key for session state

                with st.container(border=True, key=f"alert_container_search_{alert_id_search}"):
                    col_alert_info_s, col_alert_actions_s = st.columns([3, 2]) # Adjusted column ratio
                    with col_alert_info_s:
                        st.markdown(f"**Alert:** '{name_search}' <small>(Type: {type_search.capitalize()})</small>", unsafe_allow_html=True)
                        created_dt_search_fmt = datetime.strptime(created_search, '%Y-%m-%d %H:%M:%S').strftime('%d %b %Y, %I:%M %p')
                        st.caption(f"Created: {created_dt_search_fmt}")
                        last_checked_display_s_fmt = "Never"
                        if last_checked_search:
                            last_checked_display_s_fmt = datetime.strptime(last_checked_search, '%Y-%m-%d %H:%M:%S').strftime('%d %b %Y, %I:%M %p')
                        st.caption(f"Last checked for new matches: {last_checked_display_s_fmt}")
                        with st.expander("View Alert Criteria", expanded=False):
                            for k_crit, v_crit in criteria_search_display.items():
                                st.markdown(f"- {k_crit.replace('_', ' ').capitalize()}: `{v_crit}`")
                    
                    with col_alert_actions_s:
                        if st.button("🔄 Check for New Matches", key=f"check_alert_now_btn_final_{alert_id_search}", type="primary", use_container_width=True):
                            new_matches_list_s = []
                            search_from_ts_s = last_checked_search or created_search
                            
                            try: # Inner try for the specific check logic
                                if type_search == "cattle":
                                    query_new_c_s = """SELECT cfs.listing_id, uc.name, uc.breed, cfs.asking_price, cfs.location, cfs.listing_date, cfs.image_url_1
                                                     FROM cattle_for_sale cfs JOIN user_cattle uc ON cfs.cattle_id = uc.cattle_id
                                                     WHERE cfs.status = 'Available' AND cfs.listing_date > ? AND cfs.user_id != ?"""
                                    params_new_c_s = [search_from_ts_s, st.session_state.user_id]
                                    if criteria_search_display.get('breed'): query_new_c_s += " AND LOWER(uc.breed) LIKE ?"; params_new_c_s.append(f"%{criteria_search_display['breed'].lower()}%")
                                    if criteria_search_display.get('location'): query_new_c_s += " AND LOWER(cfs.location) LIKE ?"; params_new_c_s.append(f"%{criteria_search_display['location'].lower()}%")
                                    if criteria_search_display.get('max_price'): query_new_c_s += " AND cfs.asking_price <= ?"; params_new_c_s.append(criteria_search_display['max_price'])
                                    query_new_c_s += " ORDER BY cfs.listing_date DESC LIMIT 5"
                                    cursor.execute(query_new_c_s, params_new_c_s)
                                    new_matches_list_s = cursor.fetchall()
                                elif type_search == "machinery":
                                    query_new_m_s = """SELECT ml.machinery_id, ml.name, ml.type, ml.asking_price, ml.location, ml.listing_date, ml.image_url_1
                                                     FROM machinery_listings ml
                                                     WHERE ml.status = 'Available' AND ml.listing_date > ? AND ml.user_id != ?"""
                                    params_new_m_s = [search_from_ts_s, st.session_state.user_id]
                                    if criteria_search_display.get('keyword_machinery'): # Using 'keyword_machinery'
                                        term_s = f"%{criteria_search_display['keyword_machinery'].lower()}%"
                                        query_new_m_s += " AND (LOWER(ml.name) LIKE ? OR LOWER(ml.type) LIKE ? OR LOWER(ml.brand) LIKE ?)"; params_new_m_s.extend([term_s]*3)
                                    if criteria_search_display.get('condition'): query_new_m_s += " AND ml.condition = ?"; params_new_m_s.append(criteria_search_display['condition'])
                                    if criteria_search_display.get('location'): query_new_m_s += " AND LOWER(ml.location) LIKE ?"; params_new_m_s.append(f"%{criteria_search_display['location'].lower()}%")
                                    query_new_m_s += " ORDER BY ml.listing_date DESC LIMIT 5"
                                    cursor.execute(query_new_m_s, params_new_m_s)
                                    new_matches_list_s = cursor.fetchall()

                                # Store matches before DB update & toast
                                st.session_state[session_matches_key_search] = new_matches_list_s if new_matches_list_s else []
                                
                                current_check_ts_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                cursor.execute("UPDATE saved_alerts SET last_checked_at = ? WHERE alert_id = ?", (current_check_ts_update, alert_id_search))
                                conn.commit()

                                if new_matches_list_s:
                                    st.toast(f"🎉 Found {len(new_matches_list_s)} new match(es) for '{name_search}'!", icon="🎉")
                                else:
                                    st.toast(f"No new matches found for '{name_search}' this time.", icon="ℹ️")
                                
                                st.rerun()
                            except sqlite3.Error as e_check_matches_s:
                                st.error(f"Error checking for matches: {e_check_matches_s}")
                                logger.error(f"Error during 'Check for New Matches' for alert {alert_id_search}: {e_check_matches_s}", exc_info=True)


                        if st.button("🗑️ Delete Search Alert", key=f"del_search_alert_btn_final_{alert_id_search}", type="secondary", use_container_width=True):
                            try:
                                cursor.execute("DELETE FROM saved_alerts WHERE alert_id = ? AND user_id = ?", (alert_id_search, st.session_state.user_id))
                                conn.commit()
                                st.success(f"Search Alert '{name_search}' deleted.")
                                if session_matches_key_search in st.session_state: del st.session_state[session_matches_key_search]
                                st.rerun()
                            except sqlite3.Error as e_del_sa_s_err: st.error(f"Error deleting alert: {e_del_sa_s_err}")

                    # Display new matches for THIS search alert if they exist in session state
                    if session_matches_key_search in st.session_state and st.session_state[session_matches_key_search]:
                        st.markdown(f"**New Matches Found for '{name_search}':**")
                        for match_item_s_tuple in st.session_state[session_matches_key_search]:
                            item_id_s, item_name_s, item_detail1_s, item_price_s, item_loc_s, item_date_s, item_img_s = match_item_s_tuple
                            with st.container(border=True, key=f"match_item_search_disp_final_{alert_id_search}_{item_id_s}"):
                                match_c1_s_disp, match_c2_s_disp = st.columns([1,3])
                                with match_c1_s_disp: display_uploaded_image(item_img_s, caption=item_name_s, width=100)
                                with match_c2_s_disp:
                                    st.markdown(f"**{item_name_s}** ({item_detail1_s})")
                                    st.markdown(f"Price: ₹{item_price_s:,.0f} | Location: {item_loc_s}")
                                    item_date_fmt_s_disp = datetime.strptime(item_date_s, '%Y-%m-%d %H:%M:%S').strftime('%d %b %Y')
                                    st.caption(f"Listed: {item_date_fmt_s_disp}")
                                    if st.button("Go to Listing Details", key=f"goto_listing_alert_final_{type_search}_{item_id_s}_{alert_id_search}"):
                                        if type_search == 'cattle':
                                            st.session_state.expanded_cattle_listing_id = item_id_s
                                            st.session_state.current_page = "Browse Cattle"
                                        elif type_search == 'machinery':
                                            st.session_state.expanded_machinery_listing_id = item_id_s
                                            st.session_state.current_page = "Browse Machinery"
                                        st.rerun()
                        if st.button("Clear These Matches", key=f"clear_matches_search_btn_final_{alert_id_search}", type="secondary"):
                            del st.session_state[session_matches_key_search]
                            st.rerun()
                    st.markdown("---") # Separator between search alerts
    except sqlite3.Error as e_fetch_alerts_main_err_outer:
        st.error(f"Could not fetch your saved search alerts: {e_fetch_alerts_main_err_outer}")
        logger.error(f"DB error fetching saved_alerts for user {st.session_state.user_id}: {e_fetch_alerts_main_err_outer}", exc_info=True)

    # --- Section 2: Items You're Tracking (Wishlist) ---
    st.markdown("---")
    ts.subheader("❤️ Items You Are Actively Tracking (My Wishlist)")
    try:
        cursor.execute("""
            SELECT sl.saved_id, sl.listing_type, sl.original_listing_id, sl.saved_at,
                   cfs.description AS cattle_desc, cfs.asking_price AS cattle_price, cfs.image_url_1 AS cattle_img, 
                   uc.name AS cattle_name, uc.breed AS cattle_breed, uc.sex AS cattle_sex, uc.dob AS cattle_dob,
                   ml.description AS mach_desc, ml.asking_price AS mach_price, ml.image_url_1 AS mach_img, 
                   ml.name AS mach_name, ml.type AS mach_type, ml.condition AS mach_condition,
                   u_seller.username AS seller_username
            FROM user_saved_listings sl
            LEFT JOIN cattle_for_sale cfs ON sl.listing_type = 'cattle' AND sl.original_listing_id = cfs.listing_id
            LEFT JOIN user_cattle uc ON cfs.cattle_id = uc.cattle_id
            LEFT JOIN machinery_listings ml ON sl.listing_type = 'machinery' AND sl.original_listing_id = ml.machinery_id
            LEFT JOIN users u_seller ON (cfs.user_id = u_seller.user_id OR ml.user_id = u_seller.user_id)
            WHERE sl.user_id = ? ORDER BY sl.saved_at DESC
        """, (st.session_state.user_id,))
        tracked_items_data = cursor.fetchall()

        if not tracked_items_data:
            st.info("You are not currently tracking any specific items. Browse listings and click 'Mark as Interested & Track' or 'Save to Wishlist'.")
        else:
            for item_data_tracked in tracked_items_data:
                (saved_id_tracked, item_type_tracked, original_id_tracked, saved_at_str_tracked,
                 c_desc_tracked, c_price_tracked, c_img_tracked, c_name_tracked, c_breed_tracked, c_sex_tracked, c_dob_tracked,
                 m_desc_tracked, m_price_tracked, m_img_tracked, m_name_tracked, m_type_tracked, m_condition_tracked,
                 seller_username_tracked) = item_data_tracked

                with st.container(border=True, key=f"tracked_item_disp_cont_final_{saved_id_tracked}"):
                    item_name_disp_tracked, item_details_disp_tracked, item_price_disp_tracked, item_image_disp_tracked = "", "", "", None
                    col_img_tracked_disp, col_info_tracked_disp, col_actions_tracked_disp = st.columns([1,2,1])

                    if item_type_tracked == 'cattle':
                        item_name_disp_tracked = c_name_tracked or f"Cattle Listing {original_id_tracked}"
                        age_str_tracked_disp = "N/A"
                        if c_dob_tracked:
                            try: bd_t = datetime.strptime(c_dob_tracked, "%Y-%m-%d").date(); td_t = date.today(); age_d_t=td_t-bd_t; y_t=age_d_t.days//365; m_t=(age_d_t.days%365)//30; age_str_tracked_disp=f"{y_t}y {m_t}m"
                            except: pass
                        item_details_disp_tracked = f"**Breed:** {c_breed_tracked or 'N/A'} | **Sex:** {c_sex_tracked or 'N/A'} | **Age:** {age_str_tracked_disp}"
                        item_price_disp_tracked = f"₹{c_price_tracked:,.0f}" if c_price_tracked else "Price N/A"
                        item_image_disp_tracked = c_img_tracked
                        with col_info_tracked_disp: st.subheader(f"🐄 {item_name_disp_tracked}")
                    elif item_type_tracked == 'machinery':
                        item_name_disp_tracked = m_name_tracked or f"Machinery {original_id_tracked}"
                        item_details_disp_tracked = f"**Type:** {m_type_tracked or 'N/A'} | **Condition:** {m_condition_tracked or 'N/A'}"
                        item_price_disp_tracked = f"₹{m_price_tracked:,.0f}" if m_price_tracked else "Price N/A"
                        item_image_disp_tracked = m_img_tracked
                        with col_info_tracked_disp: st.subheader(f"🚜 {item_name_disp_tracked}")
                    
                    with col_img_tracked_disp:
                        display_uploaded_image(item_image_disp_tracked, caption=item_name_disp_tracked, use_container_width=True)
                    with col_info_tracked_disp:
                        st.markdown(item_details_disp_tracked); st.markdown(f"**Asking Price:** {item_price_disp_tracked}")
                        st.caption(f"Tracked since: {datetime.strptime(saved_at_str_tracked, '%Y-%m-%d %H:%M:%S').strftime('%d %b %Y')} | Seller: {seller_username_tracked or 'Unknown'}")
                        if st.button("View Original Listing Details", key=f"view_orig_listing_btn_final_{item_type_tracked}_{original_id_tracked}"):
                            if item_type_tracked == 'cattle':
                                st.session_state.expanded_cattle_listing_id = original_id_tracked
                                st.session_state.current_page = "Browse Cattle"
                            elif item_type_tracked == 'machinery':
                                st.session_state.expanded_machinery_listing_id = original_id_tracked
                                st.session_state.current_page = "Browse Machinery"
                            st.rerun()
                    with col_actions_tracked_disp:
                        if st.button("❌ Stop Tracking", key=f"remove_tracked_item_btn_final_{saved_id_tracked}", type="secondary", use_container_width=True):
                            try:
                                cursor.execute("DELETE FROM user_saved_listings WHERE saved_id = ? AND user_id = ?", (saved_id_tracked, st.session_state.user_id))
                                conn.commit(); st.toast("Item removed from your tracking list.", icon="🗑️"); st.rerun()
                            except sqlite3.Error as e_rem_track: st.error(f"Error removing: {e_rem_track}")
                    st.markdown("---")
    except sqlite3.Error as e_fetch_tracked_err_outer:
                st.error(f"Could not fetch your tracked items: {e_fetch_tracked_err_outer}")
                logger.error(f"DB error fetching user_saved_listings for user {st.session_state.user_id}: {e_fetch_tracked_err_outer}", exc_info=True)
    # No conn.close() due to @st.cache_resource for get_connection()
# --- End of "Saved Alerts" Page Content ---
# --- In your main PAGE CONTENT ROUTING section ---

elif st.session_state.current_page == "Buyer Dashboard":
    # Access Control is assumed to be handled before this block
    # and ensures st.session_state.role == "buyer"
    ts.title(f"🛍️ Buyer Dashboard - Welcome {st.session_state.username}!")
    ts.markdown("Your hub for discovering new listings and managing your saved alerts.")
    ts.markdown("---")

    conn = get_connection()
    if not conn:
        ts.error("Database connection failed.")
        st.stop() # Stop rendering this page if DB is down

    cursor = conn.cursor()

    # --- Quick Stats relevant to Buyers ---
    ts.subheader("📈 Market at a Glance")
    # FIX: st.columns for layout, not ts.columns
    col_stat_b1, col_stat_b2, col_stat_b3, col_stat_b4 = st.columns(4)
    yesterday_ts = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

    try:
        cursor.execute("SELECT COUNT(*) FROM cattle_for_sale WHERE status = 'Available' AND listing_date > ?", (yesterday_ts,))
        new_cattle_count = cursor.fetchone()[0]
        col_stat_b1.metric("🐄 New Cattle (24h)", new_cattle_count)
    except Exception as e: col_stat_b1.caption(f"Err: {e}")

    try:
        cursor.execute("SELECT COUNT(*) FROM machinery_listings WHERE status = 'Available' AND listing_date > ?", (yesterday_ts,))
        new_machinery_count = cursor.fetchone()[0]
        col_stat_b2.metric("🚜 New Machinery (24h)", new_machinery_count)
    except Exception as e: col_stat_b2.caption(f"Err: {e}")

    try:
        cursor.execute("SELECT COUNT(*) FROM saved_alerts WHERE user_id = ? AND is_active = 1", (st.session_state.user_id,))
        active_alerts_count = cursor.fetchone()[0]
        col_stat_b3.metric("🔔 Your Active Alerts", active_alerts_count)
    except Exception as e: col_stat_b3.caption(f"Err: {e}")

    try:
        cursor.execute("SELECT COUNT(*) FROM user_saved_listings WHERE user_id = ?", (st.session_state.user_id,))
        wishlist_count = cursor.fetchone()[0]
        col_stat_b4.metric("❤️ Your Wishlist Items", wishlist_count)
    except Exception as e: col_stat_b4.caption(f"Err: {e}")


    ts.markdown("---")
    ts.subheader("🚀 Quick Actions")
    # FIX: st.columns for layout, not ts.columns
    q_col_b1, q_col_b2 = st.columns(2)
    with q_col_b1:
        # FIX: st.button for navigation buttons, not ts.button
        if st.button("🔍 Browse All Cattle", use_container_width=True, key="dash_b_browse_c_main", type="primary"):
            st.session_state.current_page = "Browse Cattle"; st.rerun()
        if st.button("⚙️ Sell Your Machinery", use_container_width=True, key="dash_b_sell_m_main"):
            st.session_state.current_page = "Sell Machinery"; st.rerun()
    with q_col_b2:
        # FIX: st.button for navigation buttons, not ts.button
        if st.button("🚜 Browse All Machinery", use_container_width=True, key="dash_b_browse_m_main", type="primary"):
            st.session_state.current_page = "Browse Machinery"; st.rerun()
        if st.button("🔔 My Saved Alerts", use_container_width=True, key="dash_b_alerts_main"):
            st.session_state.current_page = "Saved Alerts"; st.rerun()
    # FIX: st.button for navigation buttons, not ts.button
    if st.button("❤️ View My Wishlist", use_container_width=True, key="dash_b_wishlist_main"): # Centered button
        st.session_state.current_page = "My Wishlist"; st.rerun()

    ts.markdown("---")
    ts.subheader("✨ Recently Added Listings")
    
    # Display a few recent cattle listings
    try:
        cursor.execute("""
            SELECT cfs.listing_id, uc.name, uc.breed, cfs.asking_price, cfs.location, cfs.image_url_1
            FROM cattle_for_sale cfs
            JOIN user_cattle uc ON cfs.cattle_id = uc.cattle_id
            WHERE cfs.status = 'Available' AND cfs.user_id != ? 
            ORDER BY cfs.listing_date DESC LIMIT 3
        """, (st.session_state.user_id,)) # Exclude buyer's own (if they could list cattle)
        recent_cattle = cursor.fetchall()
        if recent_cattle:
            ts.markdown("##### 🐄 Latest Cattle:")
            # FIX: st.columns for layout, not ts.columns
            cols_cattle = st.columns(len(recent_cattle))
            for i, cattle_item in enumerate(recent_cattle):
                list_id, name, breed, price, loc, img = cattle_item # Assumes 6 columns selected
                with cols_cattle[i]:
                    # FIX: st.container for layout, not ts.container
                    with st.container(border=True):
                        display_uploaded_image(img, caption=f"{name} ({breed})", use_container_width=True)
                        st.markdown(f"**{name or 'Cattle'}** ({breed or 'N/A'})")
                        ts.markdown(f"₹{price:,.0f} - {loc or 'N/A'}")
                        # FIX: st.button for navigation buttons, not ts.button
                        if st.button("Details", key=f"dash_view_cattle_{list_id}", type="secondary", use_container_width=True):
                            st.session_state.expanded_cattle_listing_id = list_id
                            st.session_state.expanded_machinery_listing_id = None
                            st.session_state.current_page = "Browse Cattle"
                            st.rerun()
        else:
            ts.caption("No new cattle listings to show right now.")
    except Exception as e:
        # logger.error(f"Error fetching recent cattle for buyer dashboard: {e}", exc_info=True)
        ts.caption("Could not load recent cattle listings.")

    ts.markdown("---")
    # Display a few recent machinery listings
    try:
        cursor.execute("""
            SELECT ml.machinery_id, ml.name, ml.type, ml.asking_price, ml.location, ml.image_url_1, ml.for_rent, ml.rental_price_day
            FROM machinery_listings ml
            WHERE ml.status = 'Available' AND ml.user_id != ?
            ORDER BY ml.listing_date DESC LIMIT 3
        """, (st.session_state.user_id,)) # Exclude buyer's own machinery
        recent_machinery = cursor.fetchall()
        if recent_machinery:
            ts.markdown("##### 🚜 Latest Machinery:")
            # FIX: st.columns for layout, not ts.columns
            cols_machinery = st.columns(len(recent_machinery))
            for i, mach_item in enumerate(recent_machinery):
                mach_id, name, m_type, price, loc, img, for_rent, rent_price = mach_item # Assumes 8 columns
                with cols_machinery[i]:
                    # FIX: st.container for layout, not ts.container
                    with st.container(border=True):
                        display_uploaded_image(img, caption=f"{name} ({m_type})", use_container_width=True)
                        ts.markdown(f"**{name or 'Machinery'}** ({m_type or 'N/A'})")
                        price_str_dash = f"Sale: ₹{price:,.0f}" if price and price > 0 else ""
                        if for_rent == 1:
                            rent_str_dash = f"Rent: ₹{rent_price:,.0f}/day" if rent_price and rent_price > 0 else "For Rent"
                            price_str_dash = f"{price_str_dash} | {rent_str_dash}" if price_str_dash else rent_str_dash
                        ts.markdown(f"{price_str_dash or 'Price on Req.'} - {loc or 'N/A'}")
                        # FIX: st.button for navigation buttons, not ts.button
                        if st.button("Details", key=f"dash_view_mach_{mach_id}", type="secondary", use_container_width=True):
                            st.session_state.expanded_machinery_listing_id = mach_id
                            st.session_state.expanded_cattle_listing_id = None
                            st.session_state.current_page = "Browse Machinery"
                            st.rerun()
        else:
            ts.caption("No new machinery listings to show right now.")
    except Exception as e:
        # logger.error(f"Error fetching recent machinery for buyer dashboard: {e}", exc_info=True)
        ts.caption("Could not load recent machinery listings.")

elif selected_page == "Home": # Or: elif st.session_state.current_page == "Home":
    # --- Hero Section ---
    # Using st.columns for a banner-like effect.
    # You can also use st.image directly if you prefer a full-width banner at the very top.
    ts.title(translate_text("🐄 Kamadhenu Program: Nurturing Tradition, Cultivating Future 🇮🇳", current_lang))
    ts.caption(translate_text("A comprehensive digital platform for Indian cattle breeders, rearers, and enthusiasts.", current_lang))
    st.markdown("---")
    
    col_img, col_txt = st.columns([1, 2])
    with col_img:
        display_image("home1.jpeg", use_container_width=True) # Use base name
    with col_txt:
        ts.subheader(translate_text("Empowering Farmers, Conserving Heritage, Building Resilience", current_lang))
        ts.markdown(translate_text("""
            Welcome to the digital heart of the Kamadhenu Program! We are dedicated to revitalizing Indian agriculture by:
            * 🥇 **Championing Indigenous Breeds:** Protecting, promoting, and providing comprehensive information.
            * 💡 **Innovating with Technology:** Leveraging AI and data for smarter, more efficient farming.
            * 🌱 **Fostering Sustainability:** Advocating for eco-friendly practices for long-term prosperity.
            * 📚 **Sharing Knowledge:** Offering accessible tools, guides, and a supportive community.

            *Navigate using the menu above to discover a wealth of resources.*
            """, current_lang))
        st.link_button(translate_text("Explore Indigenous Breeds ➔", current_lang), "https://en.wikipedia.org/wiki/Indigenous_cattle_breeds_of_India", type="primary")

    st.markdown("---")
    # --- Key Features Section ---
    st.header(translate_text("✨ Features at Your Fingertips", current_lang))
    st.write(translate_text("Unlock a range of tools and information designed for your success:", current_lang))

    feat_cols = st.columns(3)
    with feat_cols[0]:
        with st.container(border=True):
            st.subheader(translate_text("🧬 Breed Information & AI ID", current_lang))
            display_static_image("feature_breed_id.png", use_container_width=True) # Placeholder: images/feature_breed_id.jpg
            st.caption(translate_text("Identify breeds with AI from images. Access detailed profiles of numerous indigenous cattle, their characteristics, and utility.", current_lang))
            if st.button(translate_text("Discover Breeds", current_lang), key="home_btn_breedinfo", use_container_width=True):
                st.session_state.current_page = "Breed Info"; st.rerun()

    with feat_cols[1]:
        with st.container(border=True):
            st.subheader(translate_text("🌱 Eco-Practices & Sustainability", current_lang))
            display_static_image("feature_eco.png", use_container_width=True) # Placeholder: images/feature_eco.jpg
            st.caption(translate_text("Learn about organic farming, water conservation, manure management, biogas, and other sustainable techniques to enhance farm resilience.", current_lang))
            st.button(translate_text("Learn Eco Practices", current_lang), key="home_btn_eco")
            st.session_state.current_page = "Eco Practices"; st.rerun()

    with feat_cols[2]:
        with st.container(border=True):
            st.subheader(translate_text("❤️‍🩹 Health, Lifecycle & Diagnosis", current_lang))
            display_static_image("feature_health.png", use_container_width=True) # Placeholder: images/feature_health.jpg
            st.caption(translate_text("Get preliminary diagnosis assistance (symptom & image-based). Manage cattle through all life stages with expert guidance.", current_lang))
            if st.button(translate_text("Explore Health Tools", current_lang, key="auto_btn_36"), key="home_btn_health", use_container_width=True):
                st.session_state.current_page = "Diagnosis"; st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True) # Spacer

    feat_cols2 = st.columns(3)
    with feat_cols2[0]:
        with st.container(border=True):
            st.subheader(translate_text("🛒 Marketplace & Connections", current_lang))
            display_static_image("feature_marketplace.png", use_container_width=True) # Placeholder: images/feature_marketplace.jpg
            st.caption(translate_text("Connect with buyers and sellers for cattle, machinery, and farm products. Facilitating fair trade within the community.", current_lang))
            if st.button(translate_text("Visit Marketplace", current_lang, key="auto_btn_37"), key="home_btn_market", use_container_width=True):
                # Navigate to a relevant marketplace page, e.g., Browse Cattle or a new central Market page
                st.session_state.current_page = "Farm Products"; st.rerun()


    with feat_cols2[1]:
        with st.container(border=True):
            st.subheader(translate_text("💬 Community & Knowledge Hub", current_lang))
            display_static_image("feature_community.png", use_container_width=True) # Placeholder: images/feature_community.jpg
            st.caption(translate_text("Join discussions, ask questions, and share experiences with fellow farmers, buyers, and experts in our Community Forum.", current_lang))
            if st.button(translate_text("Join the Community", current_lang, key="auto_btn_38"), key="home_btn_forum", use_container_width=True):
                st.session_state.current_page = "Community Network"; st.rerun()

    with feat_cols2[2]:
        with st.container(border=True):
            st.subheader(translate_text("💰 Financial Guidance", current_lang))
            display_static_image("feature_finance.png", use_container_width=True) # Placeholder: images/feature_finance.jpg
            st.caption(translate_text("Explore government schemes, estimate cattle valuation, and use our loan calculator to plan your farm finances effectively.", current_lang))
            if st.button(translate_text("Financial Tools", current_lang, key="auto_btn_39"), key="home_btn_finance", use_container_width=True):
                st.session_state.current_page = "Price Trends"; st.rerun() # Price Trends has the loan calc

    st.markdown("---")

    # --- Call to Action / What's New (Optional) ---
    col_cta1, col_cta2 = st.columns(2)
    with col_cta1:
        st.subheader(translate_text("🚀 Get Started Today!", current_lang))
        if not st.session_state.logged_in:
            st.markdown(translate_text("New to the Kamadhenu Program? Register now to unlock personalized features!", current_lang))
            if st.button(translate_text("📝 Register Your Account", current_lang, key="auto_btn_40"), type="primary", use_container_width=True):
                st.session_state.current_page = "Register"; st.rerun()
            st.markdown(translate_text("Already have an account?", current_lang))
            if st.button(translate_text("👤 Login to Your Dashboard", current_lang, key="auto_btn_41"), use_container_width=True):
                st.session_state.current_page = "Login"; st.rerun()
        else:
            st.markdown(translate_text(f"Welcome back, **{st.session_state.username}**! What would you like to do today?", current_lang))
            default_dash = "Farmer Dashboard" if st.session_state.role == "farmer" else "Buyer Dashboard"
            if st.button(translate_text(f"Go to My Dashboard ➔", current_lang, key="auto_btn_42"), type="primary", use_container_width=True):
                st.session_state.current_page = default_dash; st.rerun()

    with col_cta2:
        st.subheader(translate_text("🌟 What's New / Highlights", current_lang))
        # This can be dynamic if you have a system to update it, or static for now
        display_static_image("highlight_feature.png", use_container_width=500, caption=translate_text("Highlight: New AI Skin Disease Detector!", current_lang)) # Placeholder: images/highlight_feature.jpg
        st.markdown(translate_text("""
        * **New!** Preliminary AI Skin Disease Detection now available in the 'Diagnosis' section.
        * Explore our enhanced **Community Forum** for discussions.
        * **Farmer & Buyer Dashboards** for personalized experience.
        """, current_lang))

# 2. Cattle Breed Information (from original code, minor path fix)
# --- Start of "Breed Info" Page ---
elif selected_page == "Breed Info": # Or: elif st.session_state.current_page == "Breed Info":
    ts.title("🐄 Indigenous Indian Cattle Breed Profiles")
    ts.markdown("Discover the unique characteristics, origins, and utility of India's native cattle breeds.")
    st.markdown("---")

    conn = get_connection() # Get the database connection
    if not conn:
        ts.error("Database connection failed. Cannot load breed information.")
        st.stop() # Stop further execution for this page

    cursor = conn.cursor() # Create a cursor from the connection

    cattle_breeds_from_db = [] # Initialize to an empty list
    try:
        # Fetch breed data from the database
        cursor.execute("""
            SELECT name, region, milk_yield, strength, lifespan, image_url, symbolism, scripture, research
            FROM cattle_breeds
            ORDER BY name ASC
        """) # Added ORDER BY for consistency
        rows = cursor.fetchall()
        if rows:
            cattle_breeds_from_db = [
                {
                    "name": row[0], "region": row[1], "milk_yield": row[2],
                    "strength": row[3], "lifespan": row[4],
                    "image": row[5], # This key should match what display_image expects
                                     # and your CATTLE_BREEDS_DATA static list if you merge/compare
                    "symbolism": row[6], "scripture": row[7], "research": row[8]
                }
                for row in rows
            ]
        else:
            ts.info("No breed information found in the database. Using static data as fallback (if available).")
            # Optionally, fall back to your CATTLE_BREEDS_DATA if DB is empty
            # cattle_breeds_from_db = CATTLE_BREEDS_DATA # If you want this behavior
            # For now, we'll proceed assuming if DB is empty, the list is empty.

    except sqlite3.Error as e_fetch_breeds:
        ts.error(f"Database error fetching breed information: {e_fetch_breeds}")
        logger.error(f"DB Error fetching cattle_breeds: {e_fetch_breeds}")
        # Optionally fall back to static data on error too
        # cattle_breeds_from_db = CATTLE_BREEDS_DATA
    # No conn.close() here if get_connection() is cached with @st.cache_resource

    # Use cattle_breeds_from_db for filtering and display
    # If cattle_breeds_from_db is empty, the rest of the logic will show "No breeds match"
    
    # Filters
    col1_bi, col2_bi, col3_bi = st.columns([2, 2, 1]) # Unique keys for filters
    with col1_bi:
        search_query_bi = st.text_input("🔍 Search by Breed Name:", placeholder="E.g., Sahiwal, Gir...", key="bi_search_query")
    with col2_bi:
        # Get unique regions from the data we just fetched (or static if DB failed)
        unique_regions_bi = sorted(list(set(b["region"] for b in cattle_breeds_from_db if b.get("region"))))
        selected_region_bi = st.selectbox("🌍 Filter by Region:", ["All"] + unique_regions_bi, key="bi_region_select")
    with col3_bi:
        sort_options_bi = ["Name", "Milk Yield", "Strength", "Lifespan"]
        sort_option_bi = st.selectbox("📊 Sort by:", sort_options_bi, key="bi_sort_option")

    # Apply filters
    filtered_breeds_display = cattle_breeds_from_db # Start with what we got from DB
    if search_query_bi:
        filtered_breeds_display = [b for b in filtered_breeds_display if search_query_bi.lower() in b.get("name", "").lower()]
    if selected_region_bi != "All":
        filtered_breeds_display = [b for b in filtered_breeds_display if b.get("region") == selected_region_bi]

    # Sorting
    if filtered_breeds_display: # Only sort if there's something to sort
        if sort_option_bi == "Milk Yield":
            filtered_breeds_display = sorted(filtered_breeds_display, key=lambda x: x.get("milk_yield", 0), reverse=True)
        elif sort_option_bi == "Lifespan":
            filtered_breeds_display = sorted(filtered_breeds_display, key=lambda x: x.get("lifespan", 0), reverse=True)
        elif sort_option_bi == "Strength":
            strength_order_bi = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
            filtered_breeds_display = sorted(filtered_breeds_display, key=lambda x: strength_order_bi.get(x.get("strength", "Low"), 1), reverse=True)
        else: # Default sort by Name
            filtered_breeds_display = sorted(filtered_breeds_display, key=lambda x: x.get("name", ""))

    # Display cards
    if filtered_breeds_display:
        cols_breed_display = st.columns(3) # Unique name for columns
        for i, breed_item_display in enumerate(filtered_breeds_display): # Unique loop variable
            with cols_breed_display[i % 3]:
                with st.container(border=True):
                    st.subheader(f"{breed_item_display.get('name', 'N/A')}")
                    # Assuming display_image helper correctly uses the 'images/' subfolder
                    display_image(breed_item_display.get("image", ""), caption=f"{breed_item_display.get('name', '')} ({breed_item_display.get('region', '')})")
                    ts.markdown(f"**🥛 Avg. Milk Yield:** {breed_item_display.get('milk_yield', 'N/A')} L/day")
                    ts.markdown(f"**💪 Strength/Draft:** {breed_item_display.get('strength', 'N/A')}")
                    ts.markdown(f"**⏳ Avg. Lifespan:** {breed_item_display.get('lifespan', 'N/A')} years")
                    if breed_item_display.get("symbolism"):
                        ts.markdown(f"**🔔 Symbolism:** {breed_item_display.get('symbolism')}")
                    if breed_item_display.get("scripture"):
                        ts.markdown(f"**📜 Scripture:** {breed_item_display.get('scripture')}")
                    if breed_item_display.get("research"):
                        ts.markdown(f"**🔬 Research:** {breed_item_display.get('research')}")
            
    else:
        ts.warning("⚠️ No breeds match your current filters or no breed data available.")

    ts.markdown("---")
    ts.subheader("🧠 Cultural & Scientific Insights")
    # Your tabs for Panchgavya, Breed Significance, Modern Research remain the same
    tab1_bi, tab2_bi, tab3_bi = st.tabs(["Panchgavya", "Breed Significance", "Modern Research"])
    with tab1_bi:
        ts.subheader("🧪 Panchgavya Applications")
        display_static_image(r"images\p5.jpg", use_container_width=True) # Use your helper
        ts.markdown("- **Milk**: Ayurvedic preparations\n- **Ghee**: Rituals and medicine\n- **Dung**: Organic farming\n- **Urine**: Traditional remedies\n- **Curd**: Nutritional benefits")
    with tab2_bi:
        ts.subheader("✨ Breed Significance: Symbolism & Scriptures")

        if not cattle_breeds_from_db: # Check if data was loaded
            ts.warning("No breed data available to display significance.")
        else:
            breed_names_for_select = ["--- Select a Breed ---"] + sorted([b.get("name", "Unknown") for b in cattle_breeds_from_db])
            
            selected_breed_name_for_significance = st.selectbox(
                "Select a breed to see its cultural significance:",
                options=breed_names_for_select,
                index=0, # Default to "--- Select a Breed ---"
                key="breed_significance_select"
            )

            if selected_breed_name_for_significance != "--- Select a Breed ---":
                # Find the selected breed's data
                selected_breed_data = None
                for breed_info in cattle_breeds_from_db:
                    if breed_info.get("name") == selected_breed_name_for_significance:
                        selected_breed_data = breed_info
                        break
                
                if selected_breed_data:
                    sig_col1, sig_col2 = st.columns(2)
                    with sig_col1:
                        ts.markdown(f"#### 🐂 Symbolism for {selected_breed_data.get('name')}")
                        symbolism_text = selected_breed_data.get("symbolism", "No specific symbolism information available for this breed.")
                        st.info(translate_text(symbolism_text if symbolism_text else "Not available.",current_lang))
                    
                    with sig_col2:
                        ts.markdown(f"#### 📜 Scriptural/Cultural References for {selected_breed_data.get('name')}")
                        scripture_text = selected_breed_data.get("scripture", "No specific scriptural or cultural references available for this breed.")
                        st.info(translate_text(scripture_text if scripture_text else "Not available.",current_lang))
                else:
                    ts.warning(f"Details not found for {selected_breed_name_for_significance}.")
            else:
                st.info(translate_text("Select a breed from the dropdown above to learn about its significance.",current_lang))
    # --- END MODIFIED "Breed Significance" Tab ---

    with tab3_bi: # Modern Research
        ts.subheader("🔬 Current Research Insights") # Changed title slightly
        # This part can also be made dynamic if your `research` field contains good summaries
        # For now, keeping your static example but adding a note about dynamic content
        
        ts.markdown("""
        Many indigenous Indian cattle breeds are subjects of modern scientific research focusing on:
        - **Genetic Resilience:** Studies on their adaptability to harsh climates, heat tolerance, and disease resistance (e.g., A2 milk properties).
        - **Milk Quality:** Research into the unique nutritional composition of their milk (e.g., beta-casein A2 allele).
        - **Conservation Genetics:** Efforts by institutions like NDRI (National Dairy Research Institute) and NBAGR (National Bureau of Animal Genetic Resources) to conserve and improve indigenous germplasm.
        - **Draught Power Efficiency:** Analysis of their suitability for agricultural work in various terrains.
        - **Panchagavya Research:** Scientific investigation into the properties and applications of cow-derived products in agriculture and traditional medicine (e.g., by IIT Delhi, CSIR labs).
        """)
        st.link_button("Explore NDRI Research", "https://ndri.res.in/research-development")
        st.link_button("Explore NBAGR", "http://14.139.252.116/annualreport.html#")

        st.markdown("---")
        ts.write("**To see research specific to a breed, select it from the 'Breed Significance' tab or view its card above if research details are available.**")
    
elif selected_page == "Temple Connect":
    
    ts.title("🛕 Temple/Goshala Network")
    ts.markdown("Find nearby temples and shelters supporting indigenous cattle breeds")
    ts.markdown("---")

    # Full temples list with coordinates for mapping
    temples = [
        {"name": "Shree Gopal Gau Seva Trust", "location": "Vadodara, Gujarat", "contact": "+91 98250 12345",
         "services": ["Milk Donation", "Cow Adoption", "Gau Pooja", "Gaushala Tours"], "lat": 22.3072, "lon": 73.1812},
        {"name": "Shree Ram Gau Raksha Kendra", "location": "Porbandar, Gujarat", "contact": "+91 98791 54321",
         "services": ["Panchgavya Products", "Goshala Volunteering", "Cow Shelter"], "lat": 21.6417, "lon": 69.6293},
        {"name": "Maharshi Dayanand Gau Shala", "location": "Ahmedabad, Gujarat", "contact": "+91 97234 56789",
         "services": ["Organic Fertilizers", "Cow Sponsorship", "Ayurvedic Products"], "lat": 23.0225, "lon": 72.5714},
        {"name": "Gaudham Mahatirth", "location": "Rajkot, Gujarat", "contact": "+91 99099 22334",
         "services": ["Cow Protection", "Traditional Farming", "Panchgavya Medicines"], "lat": 22.3039, "lon": 70.8022},
        {"name": "Bhaktivedanta Goshala (ISKCON)", "location": "Vrindavan, Uttar Pradesh", "contact": "+91 98371 34567",
         "services": ["Cow Care Education", "Cow Feeding", "Dairy Products"], "lat": 27.5741, "lon": 77.6960},
        {"name": "Sri Sri Goshala", "location": "Mayapur, West Bengal", "contact": "+91 98042 87654",
         "services": ["Cow Shelter", "Organic Manure", "Volunteer Program"], "lat": 23.4240, "lon": 88.3778},
        {"name": "Akhil Bhartiya Gau Seva Sangh", "location": "Nagpur, Maharashtra", "contact": "+91 98600 11223",
         "services": ["Cow Protection Awareness", "Panchagavya Medicine", "Adoption Drives"], "lat": 21.1458, "lon": 79.0882},
        {"name": "Shree Krishna Gau Raksha Trust", "location": "Chennai, Tamil Nadu", "contact": "+91 94444 22110",
         "services": ["Milk Sales", "Cow Feeding Donation", "Goshala Management"], "lat": 13.0827, "lon": 80.2707},
        {"name": "Karuna Goshala", "location": "Bangalore, Karnataka", "contact": "+91 97400 11009",
         "services": ["Rescue Cows", "Organic Products", "Volunteering"], "lat": 12.9716, "lon": 77.5946},
        {"name": "Gau Seva Dham Hospital", "location": "Pathmeda, Rajasthan", "contact": "+91 99281 34565",
         "services": ["Cow Health Services", "Panchgavya Chikitsa", "Education & Training"], "lat": 24.5937, "lon": 73.6882},
        {"name": "Om Shree Gau Shala", "location": "Haridwar, Uttarakhand", "contact": "+91 97600 77553",
         "services": ["Cow Adoption", "Spiritual Retreats", "Milk Distribution"], "lat": 29.9457, "lon": 78.1642},
        {"name": "Sri Kamadhenu Goshala", "location": "Hyderabad, Telangana", "contact": "+91 98490 45678",
         "services": ["Daily Gau Puja", "Feed a Cow Program", "Gau Grass Fund"], "lat": 17.3850, "lon": 78.4867}
    ]

    col1, col2 = st.columns(2)
    with col1:
        search = st.text_input("🔍 Search by location or name")
    with col2:
        service_filter = st.multiselect(
            "🧰 Filter by services",
            list({service for temple in temples for service in temple["services"]})
        )

    # Filter based on search and services
    filtered = temples
    if search:
        filtered = [t for t in filtered if search.lower() in t["name"].lower() or search.lower() in t["location"].lower()]
    if service_filter:
        filtered = [t for t in filtered if any(s in t["services"] for s in service_filter)]

    # Map view
    if filtered:
        df_map = pd.DataFrame([{"lat": t["lat"], "lon": t["lon"]} for t in filtered])
        st.map(df_map, zoom=4)
    else:
        st.warning("No temples found matching your filters.")

    # Display each temple entry
    for temple in filtered:
        with st.expander(f"🏛️ {temple['name']} ({temple['location']})"):
            st.write(f"📞 **Contact**: {temple['contact']}")
            st.write("🧾 **Services Offered**:")
            for service in temple["services"]:
                st.write(f"- {service}")
            st.link_button("🗺️ Get Directions", f"https://www.google.com/maps/search/?api=1&query={temple['lat']},{temple['lon']}")




# 3. Breeding Program (minor changes for DB interaction)
elif selected_page == "Breeding":
    if not st.session_state.logged_in: # Protect page
        st.warning(translate_text("You need to be logged in to access the Breeding Program.",current_lang))
        if st.button(translate_text("Login",current_lang, key="auto_btn_43")): st.session_state.current_page = "Login"; st.experimental_rerun()
    else:
        ts.title("🧬 Breeding Program Manager")
        ts.markdown("Plan, suggest, and track cattle breeding activities for desired traits.")
        st.markdown("---")
        conn = get_connection()
        if conn:
            cursor = conn.cursor()
            try: # Check tables
                cursor.execute("SELECT 1 FROM breeding_pairs LIMIT 1;")
                cursor.execute("SELECT 1 FROM offspring_data LIMIT 1;")
            except sqlite3.OperationalError:
                 st.error(translate_text("Database tables (breeding_pairs, offspring_data) not found or schema incorrect.",current_lang))
                 conn = None # Prevent further operations

            if conn:
                col1, col2 = st.columns(2)
                with col1:
                    ts.subheader("Suggest New Pairing")
                    # TODO: Populate cattle_1 and cattle_2 from user's herd if "My Herd" is used
                    # For now, keeping as text input
                    cattle_1 = st.text_input(translate_text("Name/ID of Cattle 1 (from your herd or general):",current_lang))
                    cattle_2 = st.text_input(translate_text("Name/ID of Cattle 2 (from your herd or general):",current_lang))
                    goal = st.selectbox(translate_text("Select Primary Breeding Goal",current_lang), [translate_text("High Milk Yield",current_lang), translate_text("Disease Resistance",current_lang),translate_text( "Drought Tolerance",current_lang),translate_text( "Breed Purity",current_lang), translate_text("Temperament",current_lang), translate_text("Dual Purpose (Milk & Draft)",current_lang)])

                    if st.button(translate_text("Suggest Pair",current_lang, key="auto_btn_44"), type="primary"):
                        if cattle_1 and cattle_2 and cattle_1.strip().lower() != cattle_2.strip().lower():
                            try:
                                genetic_score = random.randint(55, 95)
                                status = "Recommended" if genetic_score > 75 else ("Consider" if genetic_score > 60 else "Evaluate Carefully")
                                notes = f"Goal: {goal}. Est. Compatibility: {genetic_score}%. "
                                if genetic_score < 65: notes += "Potential mismatch in some traits."

                                cursor.execute("""
                                    INSERT INTO breeding_pairs (cattle_1, cattle_2, goal, genetic_score, status, notes)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                """, (cattle_1.strip(), cattle_2.strip(), goal, genetic_score, status, notes))
                                conn.commit() # Commit
                                st.success(translate_text(f"Pairing suggestion logged for {cattle_1} & {cattle_2}.",current_lang))
                                st.info(translate_text(f"Goal: {goal}, Score: {genetic_score}%, Status: {status}",current_lang))
                            except sqlite3.Error as e: st.error(f"Database error: {e}")
                            except Exception as e: st.error(f"Unexpected error: {e}")
                        else: st.error(translate_text("Please enter two different, non-empty cattle names/IDs.",current_lang))

                with col2:
                    ts.subheader("Recent Breeding Records")
                    # TODO: Filter records by user_id if breeding pairs become user-specific
                    tab1, tab2 = st.tabs(("📈 Suggestions Log", "🍼 Offspring Records"))
                    with tab1:
                        try:
                            cursor.execute("SELECT cattle_1, cattle_2, goal, genetic_score, status, timestamp FROM breeding_pairs ORDER BY timestamp DESC LIMIT 10")
                            pairs = cursor.fetchall()
                            if pairs:
                                df_pairs = pd.DataFrame(pairs, columns=["Cattle 1", "Cattle 2", "Goal", "Score", "Status", "Timestamp"])
                                df_pairs['Timestamp'] = pd.to_datetime(df_pairs['Timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
                                st.dataframe(df_pairs, use_container_width=True, hide_index=True)
                            else: st.info(translate_text("No breeding suggestions recorded yet.",current_lang))
                        except sqlite3.Error as e: st.warning(f"Could not fetch breeding suggestions: {e}")
                    with tab2:
                        try:
                            cursor.execute("SELECT parent_1, parent_2, offspring_id, breed, dob, sex FROM offspring_data ORDER BY timestamp DESC LIMIT 10")
                            offspring = cursor.fetchall()
                            if offspring:
                                df_offspring = pd.DataFrame(offspring, columns=["Parent 1", "Parent 2", "Offspring ID", "Breed", "DOB", "Sex"])
                                st.dataframe(df_offspring, use_container_width=True, hide_index=True)
                            else: st.info(translate_text("No offspring records yet.",current_lang))
                        except sqlite3.Error as e: st.info(translate_text(f"Offspring data not found or error: {e}",current_lang))
        else:
            st.error("Database connection failed.")


# 4. Eco-Friendly Practices (from original code - no changes needed other than DB context)
elif selected_page == "Eco Practices":
    # --- START OF THE ECO PRACTICES CODE (INDENTED ONE LEVEL FROM elif) ---
    ts.title(ECO_PAGE_TITLE_ENGLISH)
    ts.markdown(ECO_PAGE_DESC_ENGLISH)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    # --- ECO-FRIENDLY PRACTICES ---
    with col1:
        # Organic Farming
        ts.subheader(ORG_FARMING_SUBHEADER_ENGLISH)
        with st.expander(translate_text(ORG_FARMING_DETAILS_TITLE_ENGLISH, current_lang), expanded=False):
            st.markdown(translate_text(ORG_FARMING_DESC_ENGLISH, current_lang))

            # --- ADDING THE "WATCH VIDEO" BUTTON HERE ---
            if ORG_FARMING_SUBHEADER_ENGLISH in VIDEO_URLS:
                # Use a unique key for each button to avoid Streamlit errors
                button_key_organic = f"show_organic_video_{current_lang}"
                if st.button(translate_text("Watch Video", current_lang, key="auto_btn_45"), key=button_key_organic):
                    st.session_state[button_key_organic + "_clicked"] = True # Set state to show video

                # Display the video only if the button was clicked
                if st.session_state.get(button_key_organic + "_clicked", False):
                    st.video(VIDEO_URLS[ORG_FARMING_SUBHEADER_ENGLISH])
                    # Optionally, add a button to hide the video again
                    if st.button(translate_text("Hide Video", current_lang, key="auto_btn_46"), key=f"hide_organic_video_{current_lang}"):
                        st.session_state[button_key_organic + "_clicked"] = False # Reset state to hide video
                        st.rerun() # Rerun to update the display


        # Crop Rotation
        ts.subheader(CROP_ROTATION_SUBHEADER_ENGLISH)
        with st.expander(translate_text(CROP_ROTATION_DETAILS_TITLE_ENGLISH, current_lang), expanded=False):
            st.markdown(translate_text(CROP_ROTATION_DESC_ENGLISH, current_lang))
            if CROP_ROTATION_SUBHEADER_ENGLISH in VIDEO_URLS:
                button_key_crop_rotation = f"show_crop_rotation_video_{current_lang}"
                if st.button(translate_text("Watch Video", current_lang, key="auto_btn_47"), key=button_key_crop_rotation):
                    st.session_state[button_key_crop_rotation + "_clicked"] = True
                if st.session_state.get(button_key_crop_rotation + "_clicked", False):
                    st.video(VIDEO_URLS[CROP_ROTATION_SUBHEADER_ENGLISH])
                    if st.button(translate_text("Hide Video", current_lang, key="auto_btn_48"), key=f"hide_crop_rotation_video_{current_lang}"):
                        st.session_state[button_key_crop_rotation + "_clicked"] = False
                        st.rerun()


        # Water Conservation
        ts.subheader(WATER_CONS_SUBHEADER_ENGLISH)
        with st.expander(translate_text(WATER_CONS_DETAILS_TITLE_ENGLISH, current_lang), expanded=False):
            st.markdown(translate_text(WATER_CONS_DESC_ENGLISH, current_lang))
            # Example of more specific video for a sub-topic (Drip Irrigation)
            if "Drip Irrigation" in VIDEO_URLS: # Check for the specific Drip Irrigation key
                st.subheader(translate_text("Learn Drip Irrigation", current_lang)) # Display a subheader for this specific video
                button_key_drip_irrigation = f"show_drip_irrigation_video_{current_lang}"
                if st.button(translate_text("Watch Video", current_lang, key="auto_btn_49"), key=button_key_drip_irrigation):
                    st.session_state[button_key_drip_irrigation + "_clicked"] = True
                if st.session_state.get(button_key_drip_irrigation + "_clicked", False):
                    st.video(VIDEO_URLS["Drip Irrigation"])
                    if st.button(translate_text("Hide Video", current_lang, key="auto_btn_50"), key=f"hide_drip_irrigation_video_{current_lang}"):
                        st.session_state[button_key_drip_irrigation + "_clicked"] = False
                        st.rerun()


    with col2:
        # Integrated Pest Management
        ts.subheader(IPM_SUBHEADER_ENGLISH)
        with st.expander(translate_text(IPM_DETAILS_TITLE_ENGLISH, current_lang), expanded=False):
            st.markdown(translate_text(IPM_DESC_ENGLISH, current_lang))
            if IPM_SUBHEADER_ENGLISH in VIDEO_URLS:
                button_key_ipm = f"show_ipm_video_{current_lang}"
                if st.button(translate_text("Watch Video", current_lang, key="auto_btn_51"), key=button_key_ipm):
                    st.session_state[button_key_ipm + "_clicked"] = True
                if st.session_state.get(button_key_ipm + "_clicked", False):
                    st.video(VIDEO_URLS[IPM_SUBHEADER_ENGLISH])
                    if st.button(translate_text("Hide Video", current_lang, key="auto_btn_52"), key=f"hide_ipm_video_{current_lang}"):
                        st.session_state[button_key_ipm + "_clicked"] = False
                        st.rerun()


        # Manure Management
        ts.subheader(MANURE_MGMT_SUBHEADER_ENGLISH)
        with st.expander(translate_text(MANURE_MGMT_DETAILS_TITLE_ENGLISH, current_lang), expanded=False):
            st.markdown(translate_text(MANURE_MGMT_DESC_ENGLISH, current_lang))
            if MANURE_MGMT_SUBHEADER_ENGLISH in VIDEO_URLS:
                button_key_manure_mgmt = f"show_manure_mgmt_video_{current_lang}"
                if st.button(translate_text("Watch Video", current_lang, key="auto_btn_53"), key=button_key_manure_mgmt):
                    st.session_state[button_key_manure_mgmt + "_clicked"] = True
                if st.session_state.get(button_key_manure_mgmt + "_clicked", False):
                    st.video(VIDEO_URLS[MANURE_MGMT_SUBHEADER_ENGLISH])
                    if st.button(translate_text("Hide Video", current_lang, key="auto_btn_54"), key=f"hide_manure_mgmt_video_{current_lang}"):
                        st.session_state[button_key_manure_mgmt + "_clicked"] = False
                        st.rerun()


        # Vermicomposting
        ts.subheader(VERMICOMPOSTING_SUBHEADER_ENGLISH)
        with st.expander(translate_text(VERMICOMPOSTING_DETAILS_TITLE_ENGLISH, current_lang), expanded=False):
            st.markdown(translate_text(VERMICOMPOSTING_DESC_ENGLISH, current_lang))
            if VERMICOMPOSTING_SUBHEADER_ENGLISH in VIDEO_URLS:
                button_key_vermicomposting = f"show_vermicomposting_video_{current_lang}"
                if st.button(translate_text("Watch Video", current_lang, key="auto_btn_55"), key=button_key_vermicomposting):
                    st.session_state[button_key_vermicomposting + "_clicked"] = True
                if st.session_state.get(button_key_vermicomposting + "_clicked", False):
                    st.video(VIDEO_URLS[VERMICOMPOSTING_SUBHEADER_ENGLISH])
                    if st.button(translate_text("Hide Video", current_lang, key="auto_btn_56"), key=f"hide_vermicomposting_video_{current_lang}"):
                        st.session_state[button_key_vermicomposting + "_clicked"] = False
                        st.rerun()


    with col3:
        # Biogas Production
        ts.subheader(BIOGAS_SUBHEADER_ENGLISH)
        with st.expander(translate_text(BIOGAS_DETAILS_TITLE_ENGLISH, current_lang), expanded=False):
            st.markdown(translate_text(BIOGAS_DESC_ENGLISH, current_lang))
            if BIOGAS_SUBHEADER_ENGLISH in VIDEO_URLS:
                button_key_biogas = f"show_biogas_video_{current_lang}"
                if st.button(translate_text("Watch Video", current_lang, key="auto_btn_57"), key=button_key_biogas):
                    st.session_state[button_key_biogas + "_clicked"] = True
                if st.session_state.get(button_key_biogas + "_clicked", False):
                    st.video(VIDEO_URLS[BIOGAS_SUBHEADER_ENGLISH])
                    if st.button(translate_text("Hide Video", current_lang, key="auto_btn_58"), key=f"hide_biogas_video_{current_lang}"):
                        st.session_state[button_key_biogas + "_clicked"] = False
                        st.rerun()


        # Agroforestry
        ts.subheader(AGROFORESTRY_SUBHEADER_ENGLISH)
        with st.expander(translate_text(AGROFORESTRY_DETAILS_TITLE_ENGLISH, current_lang), expanded=False):
            st.markdown(translate_text(AGROFORESTRY_DESC_ENGLISH, current_lang))
            if AGROFORESTRY_SUBHEADER_ENGLISH in VIDEO_URLS:
                button_key_agroforestry = f"show_agroforestry_video_{current_lang}"
                if st.button(translate_text("Watch Video", current_lang, key="auto_btn_59"), key=button_key_agroforestry):
                    st.session_state[button_key_agroforestry + "_clicked"] = True
                if st.session_state.get(button_key_agroforestry + "_clicked", False):
                    st.video(VIDEO_URLS[AGROFORESTRY_SUBHEADER_ENGLISH])
                    if st.button(translate_text("Hide Video", current_lang, key="auto_btn_60"), key=f"hide_agroforestry_video_{current_lang}"):
                        st.session_state[button_key_agroforestry + "_clicked"] = False
                        st.rerun()


        # Rotational Grazing
        ts.subheader(ROTATIONAL_GRAZING_SUBHEADER_ENGLISH)
        with st.expander(translate_text(ROTATIONAL_GRAZING_DETAILS_TITLE_ENGLISH, current_lang), expanded=False):
            st.markdown(translate_text(ROTATIONAL_GRAZING_DESC_ENGLISH, current_lang))
            if ROTATIONAL_GRAZING_SUBHEADER_ENGLISH in VIDEO_URLS:
                button_key_rotational_grazing = f"show_rotational_grazing_video_{current_lang}"
                if st.button(translate_text("Watch Video", current_lang, key="auto_btn_61"), key=button_key_rotational_grazing):
                    st.session_state[button_key_rotational_grazing + "_clicked"] = True
                if st.session_state.get(button_key_rotational_grazing + "_clicked", False):
                    st.video(VIDEO_URLS[ROTATIONAL_GRAZING_SUBHEADER_ENGLISH])
                    if st.button(translate_text("Hide Video", current_lang, key="auto_btn_62"), key=f"hide_rotational_grazing_video_{current_lang}"):
                        st.session_state[button_key_rotational_grazing + "_clicked"] = False
                        st.rerun()


    st.markdown("---")

    # --- INDIGENOUS COW PRODUCTS & PANCHAGAVYA ---
    ts.title(COW_PRODUCTS_TITLE_ENGLISH)
    ts.markdown(COW_PRODUCTS_DESC_ENGLISH)
    st.markdown("---")

    col1_cow, col2_cow, col3_cow = st.columns(3)

    with col1_cow:
        # Milk
        ts.subheader(MILK_SUBHEADER_ENGLISH)
        with st.expander(translate_text(MILK_DETAILS_TITLE_ENGLISH, current_lang), expanded=False):
            st.markdown(translate_text(MILK_DESC_ENGLISH, current_lang))

        # Ghee
        ts.subheader(GHEE_SUBHEADER_ENGLISH)
        with st.expander(translate_text(GHEE_DETAILS_TITLE_ENGLISH, current_lang), expanded=False):
            st.markdown(translate_text(GHEE_DESC_ENGLISH, current_lang))

    with col2_cow:
        # Dung
        ts.subheader(DUNG_SUBHEADER_ENGLISH)
        with st.expander(translate_text(DUNG_DETAILS_TITLE_ENGLISH, current_lang), expanded=False):
            st.markdown(translate_text(DUNG_DESC_ENGLISH, current_lang))

        # Curd/Yogurt
        ts.subheader(CURD_SUBHEADER_ENGLISH)
        with st.expander(translate_text(CURD_DETAILS_TITLE_ENGLISH, current_lang), expanded=False):
            st.markdown(translate_text(CURD_DESC_ENGLISH, current_lang))

    with col3_cow:
        # Urine
        ts.subheader(URINE_SUBHEADER_ENGLISH)
        with st.expander(translate_text(URINE_DETAILS_TITLE_ENGLISH, current_lang), expanded=False):
            st.markdown(translate_text(URINE_DESC_ENGLISH, current_lang))
            st.caption(translate_text(URINE_NOTE_ENGLISH, current_lang))

        # Panchagavya
        ts.subheader(PANCHAGAVYA_SUBHEADER_ENGLISH)
        with st.expander(translate_text(PANCHAGAVYA_DETAILS_TITLE_ENGLISH, current_lang), expanded=False):
            st.markdown(translate_text(PANCHAGAVYA_DESC_ENGLISH, current_lang))
            if PANCHAGAVYA_SUBHEADER_ENGLISH in VIDEO_URLS:
                # This was previously just a subheader; now it's a button
                button_key_panchagavya = f"show_panchagavya_video_{current_lang}"
                if st.button(translate_text("Watch Video on Panchagavya", current_lang, key="auto_btn_63"), key=button_key_panchagavya):
                    st.session_state[button_key_panchagavya + "_clicked"] = True
                if st.session_state.get(button_key_panchagavya + "_clicked", False):
                    st.video(VIDEO_URLS[PANCHAGAVYA_SUBHEADER_ENGLISH])
                    if st.button(translate_text("Hide Video", current_lang, key="auto_btn_64"), key=f"hide_panchagavya_video_{current_lang}"):
                        st.session_state[button_key_panchagavya + "_clicked"] = False
                        st.rerun() # Rerun to hide the video


    # Other Value-Added Products
    st.markdown("---")
    ts.subheader(OTHER_PRODUCTS_SUBHEADER_ENGLISH)
    ts.markdown(OTHER_PRODUCTS_DESC_ENGLISH)
    st.markdown("---")

    # General Info Message
    st.info(translate_text(INFO_MESSAGE_ENGLISH, current_lang))
    st.markdown("---")

    # --- TOOLS FOR SUSTAINABILITY ASSESSMENT (Calculators) ---
    ts.header(TOOLS_HEADER_ENGLISH)
    col1, col2 = st.columns(2)

    with col1:
        with st.expander(translate_text(CARBON_ESTIMATOR_TITLE_ENGLISH, current_lang)):
            st.markdown(translate_text(CARBON_ESTIMATOR_DESC_ENGLISH, current_lang))
            fuel_usage = st.number_input(translate_text(FUEL_USAGE_LABEL_ENGLISH, current_lang), min_value=0.0, step=10.0)
            fertilizer_usage = st.number_input(translate_text(FERTILIZER_USAGE_LABEL_ENGLISH, current_lang), min_value=0.0, step=5.0)
            livestock_count = st.number_input(translate_text(LIVESTOCK_COUNT_LABEL_ENGLISH, current_lang), min_value=0, step=1)
            rice_paddy_area = st.number_input(translate_text(RICE_PADDY_AREA_LABEL_ENGLISH, current_lang), min_value=0.0, step=0.1)

            if st.button(translate_text(ESTIMATE_FOOTPRINT_BUTTON_ENGLISH, current_lang, key="auto_btn_65")):
                fuel_emission = fuel_usage * 2.68
                fertilizer_emission = fertilizer_usage * 4.5
                livestock_emission = livestock_count * (1800 / 12)
                rice_emission = rice_paddy_area * 50
                total_emissions = fuel_emission + fertilizer_emission + livestock_emission + rice_emission
                st.success(translate_text(ESTIMATED_FOOTPRINT_SUCCESS_ENGLISH.format(total_emissions=total_emissions), current_lang))
                st.caption(translate_text(CARBON_NOTE_ENGLISH, current_lang))

    with col2:
        with st.expander(translate_text(WATER_CALCULATOR_TITLE_ENGLISH, current_lang)):
            st.markdown(translate_text(WATER_CALCULATOR_DESC_ENGLISH, current_lang))
            field_size = st.number_input(translate_text(FIELD_SIZE_LABEL_ENGLISH, current_lang), min_value=0.0, step=0.5)
            irrigation_per_acre = st.number_input(translate_text(IRRIGATION_DEPTH_LABEL_ENGLISH, current_lang), min_value=0.0, step=1.0, value=5.0)
            days_irrigated = st.slider(translate_text(DAYS_IRRIGATED_LABEL_ENGLISH, current_lang), 1, 31, 20)

            if st.button(translate_text(ESTIMATE_WATER_USAGE_BUTTON_ENGLISH, current_lang, key="auto_btn_66")):
                liters_per_acre_per_day = 4046.86 * (irrigation_per_acre / 1000) * 1000
                monthly_water_usage = field_size * liters_per_acre_per_day * days_irrigated
                st.success(translate_text(ESTIMATED_WATER_USAGE_SUCCESS_ENGLISH.format(monthly_water_usage=monthly_water_usage), current_lang))
                st.caption(translate_text(WATER_NOTE_ENGLISH.format(irrigation_per_acre=irrigation_per_acre), current_lang))


# 5. Identify Breed (Integrated Roboflow logic - no changes needed here)
elif selected_page == "Identify Breed":
    st.title("📸 AI Cattle Breed Identification")
    # ... (your existing Identify Breed content - should work as is if Roboflow model loads) ...
    st.markdown("Upload a clear image of a cow for AI identification.")
    st.markdown("---")
    if not roboflow_model:
        st.error("Roboflow model failed to load. Breed Identification unavailable.", icon="🚫")
    else:
        uploaded_file = st.file_uploader("Choose an image (JPG, PNG)...", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key="breed_id_uploader")
        if uploaded_file:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Uploaded Image")
                st.image(uploaded_file, use_container_width=True)
                img_bytes = uploaded_file.read()
            with col2:
                st.subheader("Analysis Results")
                temp_image_path = None
                try:
                    with st.spinner("🔎 Analyzing image..."):
                        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        image_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                        temp_image_path = f"temp_{uuid.uuid4()}.jpg"; pil_image.save(temp_image_path)
                        logger.info(f"Temp image saved: {temp_image_path}")
                        result = roboflow_model.predict(temp_image_path, confidence=CONFIDENCE_THRESHOLD, overlap=OVERLAP_THRESHOLD).json()
                        logger.info("Prediction completed.")
                        if "error" in result:
                            st.error(f"Roboflow prediction failed: {result.get('error', 'Unknown')}"); predictions = []
                        else: predictions = result.get("predictions", [])

                        if not predictions: st.warning("No objects identified.")
                        else:
                            logger.info(f"Found {len(predictions)} predictions.")
                            labels, xyxys, confidences, detected_objects_for_response = [], [], [], []
                            for item in predictions:
                                xc, yc, w, h = item["x"], item["y"], item["width"], item["height"]
                                conf, lbl = item["confidence"], item["class"]
                                xmin, ymin, xmax, ymax = xc-w/2, yc-h/2, xc+w/2, yc+h/2
                                xyxys.append([xmin, ymin, xmax, ymax]); confidences.append(conf)
                                formatted_label = f"{lbl} ({conf * 100:.1f}%)"; labels.append(formatted_label)
                                detected_objects_for_response.append({"label": lbl, "confidence": conf})
                            detections = sv.Detections(xyxy=np.array(xyxys), confidence=np.array(confidences), class_id=np.array(range(len(labels))))
                            box_annotator = sv.BoxAnnotator(thickness=2)
                            label_annotator = sv.LabelAnnotator()
                            annotated_image_with_boxes = box_annotator.annotate(scene=image_cv2.copy(), detections=detections)
                            final_annotated_image = label_annotator.annotate(scene=annotated_image_with_boxes, detections=detections, labels=labels)
                            logger.info("Image annotation completed.")
                            st.image(final_annotated_image, channels="BGR", caption="Analysis Visualization", use_container_width=True)
                            st.write("**Detected:**")
                            if detected_objects_for_response:
                                for obj_info in detected_objects_for_response:
                                     display_text = f"- **{obj_info.get('label', 'Unknown')}**"
                                     if obj_info.get('confidence'): display_text += f" (Confidence: {obj_info['confidence']*100:.1f}%)"
                                     st.success(display_text)
                except Exception as e:
                    st.error(f"An error occurred during image analysis: {e}")
                    logger.error(f"Error (Identify Breed): {e}\n{traceback.format_exc()}")
                finally:
                    if temp_image_path and os.path.exists(temp_image_path):
                        try: os.remove(temp_image_path); logger.info(f"Temp file deleted: {temp_image_path}")
                        except Exception as del_err: logger.error(f"Error deleting temp file {temp_image_path}: {del_err}")
        else: st.info(translate_text("Upload a clear image file (JPG, PNG) containing cattle to begin identification.",current_lang))


# 6. Chatbot (from original code - minor error handling improvements)
elif selected_page == "Chatbot":
    st.title("🧑‍🌾 Kamadhenu AI Assistant")
    # ... (your existing Chatbot content with robust response handling) ...
    st.markdown("Ask questions about indigenous breeds, farming practices, health, schemes, etc.")
    st.markdown("---")
    if not gemini_model:
        st.error("Chatbot unavailable: Google API Key/Model issue.", icon="🚫")
    else:
        try:
            translator = Translator()
            if "messages" not in st.session_state: st.session_state.messages = []
            if "chat_language" not in st.session_state: st.session_state.chat_language = "en"

            language_options = {"English": "en", "हिन्दी (Hindi)": "hi", "తెలుగు (Telugu)": "te", "தமிழ் (Tamil)": "ta", "ગુજરાતી (Gujarati)": "gu", "ਪੰਜਾਬੀ (Punjabi)": "pa"}
            lang_keys = list(language_options.keys()); lang_values = list(language_options.values())
            current_lang_index = lang_values.index(st.session_state.chat_language) if st.session_state.chat_language in lang_values else 0
            selected_language_name = st.selectbox("Choose interaction language:", lang_keys, index=current_lang_index, key="chat_lang_select")
            st.session_state.chat_language = language_options[selected_language_name]
            lang_code = st.session_state.chat_language

            for message in st.session_state.messages:
                with st.chat_message(message["role"]): st.markdown(message["content"])

            if prompt := st.chat_input(f"Ask your question in {selected_language_name}..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    with st.spinner(f"Thinking in {selected_language_name}..."):
                        try:
                            prompt_en = prompt
                            if lang_code != 'en':
                                try: prompt_en = translator.translate(prompt, src=lang_code, dest='en').text
                                except Exception as trans_in_err: st.warning(f"Translation error: {trans_in_err}", icon="⚠️"); # Fallback
                            contextual_prompt = f"""
                            You are 'Kamadhenu Sahayak', an AI assistant for Indian farmers and cattle rearers. Focus specifically on:
                            1. Indigenous Indian cattle breeds (like Gir, Sahiwal, Ongole, Tharparkar, Kankrej, Rathi, Hallikar, etc.): Their care, characteristics, milk yield, draft power, climate suitability, and conservation status.
                            2. Sustainable & Eco-Friendly Farming Practices relevant to India, especially those involving cattle: Manure management (composting, biogas), rotational grazing, water conservation for livestock, agroforestry/silvopasture for fodder and shade, organic farming principles for fodder crops.
                            3. Common Cattle Diseases in India: Recognizing symptoms, basic first aid/preventive measures (e.g., vaccination schedules, deworming), but **always strongly emphasize consulting a qualified veterinarian** for actual diagnosis and treatment. Do not provide specific drug dosages. Mention diseases like FMD, HS, BQ, Mastitis, Scours, Bloat.
                            4. Indian Government Schemes for Agriculture & Animal Husbandry.
                            5. General cattle lifecycle management.
                            6. Basic factors affecting cattle price/valuation.
                            Answer the user question concisely, helpfully, in a friendly tone. If unrelated, politely state your specialization.
                            User question (potentially translated): {prompt_en}
                            Respond *only* in {selected_language_name}. Use bullet points if appropriate.
                            """
                            response = gemini_model.generate_content(contextual_prompt)
                            response_text_en = ""
                            try:
                                if hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'content') and hasattr(response.candidates[0].content, 'parts') and response.candidates[0].content.parts:
                                    response_text_en = response.candidates[0].content.parts[0].text
                                else:
                                    block_reason_msg = "Unknown reason."
                                    if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason'): block_reason_msg = f"Block Reason: {response.prompt_feedback.block_reason}."
                                    elif hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'finish_reason'): block_reason_msg = f"Finish Reason: {response.candidates[0].finish_reason}."
                                    st.warning(f"AI response may be empty/blocked. {block_reason_msg}", icon="⚠️"); response_text_en = "I apologize, but I couldn't generate a complete response. Try rephrasing?"
                            except ValueError as ve: st.error(f"Error processing AI response (blocked content?): {ve}"); response_text_en = "Issue processing response (content filters?). Try rephrasing."
                            except Exception as e_resp: st.error(f"Unexpected error processing AI response: {e_resp}"); response_text_en = "Internal error processing response."

                            final_response_text = response_text_en
                            if lang_code != 'en' and response_text_en and "I apologize" not in response_text_en and "I encountered an issue" not in response_text_en:
                                 try: final_response_text = translator.translate(response_text_en, src='en', dest=lang_code).text
                                 except Exception as trans_err: st.error(f"Translation error: {trans_err}"); final_response_text = f"(Translation Error) {response_text_en}"
                            elif lang_code != 'en' and ("I apologize" in response_text_en or "I encountered an issue" in response_text_en): # Try translating error messages
                                try: final_response_text = translator.translate(response_text_en, src='en', dest=lang_code).text
                                except: pass # Keep English error if translation of error fails

                            message_placeholder.markdown(final_response_text)
                            st.session_state.messages.append({"role": "assistant", "content": final_response_text})
                        except Exception as e:
                            st.error(f"Error generating response: {e}")
                            error_msg = f"Sorry, error processing request in {selected_language_name}."
                            try:
                                if lang_code != 'en': error_msg = translator.translate(error_msg, src='en', dest=lang_code).text
                            except: pass
                            message_placeholder.markdown(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        except Exception as e: st.error(f"Chatbot initialization failed: {e}.")

# 7. Price Trends & Calculator
elif selected_page == "Price Trends":
    ts.title("📈 Cattle Price Trends & Valuation Estimator")
    ts.markdown("Analyze illustrative historical price data, estimate cattle value, and explore financial tools.") # Updated description
    st.markdown("---")

    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        # Section for Price Trends Chart
        ts.subheader("📈 Historical Average Price Trends (Illustrative Data)")
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='price_trends'")
            if cursor.fetchone():
                # Query for more granular data if available (year, month, breed, region)
                # For simplicity, let's assume a simplified 'year', 'average_price' structure for now
                # If your DB has the more granular structure, adjust the query and DataFrame creation
                cursor.execute("SELECT year, AVG(average_price) as avg_price FROM price_trends GROUP BY year ORDER BY year ASC")
                data = cursor.fetchall()
                if data:
                    df_trends = pd.DataFrame(data, columns=["Year", "Average Price (INR)"])
                    df_trends = df_trends.set_index("Year")
                    st.line_chart(df_trends)
                    if len(df_trends) > 1:
                        latest_price = df_trends["Average Price (INR)"].iloc[-1]
                        previous_price = df_trends["Average Price (INR)"].iloc[-2]
                        price_change = latest_price - previous_price
                        st.metric(label=translate_text("Latest Avg Price (Overall)",current_lang), value=f"₹{latest_price:,.0f}", delta=f"₹{price_change:,.0f} vs Previous Year")
                    elif len(df_trends) == 1:
                         st.metric(label=translate_text("Latest Avg Price (Overall)",current_lang), value=f"₹{df_trends['Average Price (INR)'].iloc[-1]:,.0f}")
                else:
                    st.info(translate_text("No historical price data found in the database to display trends.",current_lang))
            else:
                 st.warning(translate_text("Database table 'price_trends' not found.",current_lang))
        except sqlite3.Error as e:
            st.error(translate_text(f"Database error fetching price trends: {e}",current_lang))

        st.markdown("---")
        # Section for Price Calculator
        ts.subheader("📊 Cattle Valuation Estimator")
        ts.caption("Provides a rough estimate. Actual market value depends on many local factors.")

        col1_val, col2_val, col3_val = st.columns(3)
        with col1_val:
            breed_list_val = sorted([b["name"] for b in CATTLE_BREEDS_DATA]) + ["Murrah (Buffalo)", "Crossbred", "Other"]
            breed_val = st.selectbox(translate_text("Select Breed",current_lang), breed_list_val, key="calc_breed_val")
        with col2_val:
            age_val = st.number_input(translate_text("Age (Years)",current_lang), min_value=0.5, max_value=25.0, value=4.0, step=0.5, key="calc_age_val")
        with col3_val:
             weight_val = st.number_input(translate_text("Approx. Weight (Kg)",current_lang), min_value=50, max_value=1200, value=350, step=10, key="calc_weight_val")

        col4_val, col5_val, col6_val = st.columns(3)
        with col4_val:
            milk_yield_val = st.number_input(translate_text("Avg. Daily Milk Yield (Liters)",current_lang), min_value=0.0, max_value=50.0, value=8.0, step=0.5, key="calc_milk_val", help="Enter 0 if not applicable/male")
        with col5_val:
             health_status_val = st.selectbox(translate_text("Overall Health Condition",current_lang), ["Excellent", "Good", "Fair", "Poor"], key="calc_health_val")
        with col6_val:
            is_pregnant_val = st.checkbox(translate_text("Currently Pregnant?",current_lang), key="calc_pregnant_val")

        if st.button(translate_text("Estimate Valuation",current_lang, key="auto_btn_67"), type="primary", key="btn_estimate"):
            base_price = 30000
            breed_factors = {
                "Gir": 1.8, "Sahiwal": 1.9, "Red Sindhi": 1.7, "Tharparkar": 1.6,
                "Ongole": 1.5, "Kankrej": 1.6, "Rathi": 1.5, "Murrah (Buffalo)": 2.0,
                "Crossbred": 1.2, "Punganur": 1.0, "Amrit Mahal": 1.3, "Hallikar": 1.4,
                "Deoni": 1.4, "Krishna Valley": 1.4, "Malnad Gidda": 1.1, "Other": 0.9
            }
            base_price *= breed_factors.get(breed_val, 1.0) # Use breed_val
            if 2.5 <= age_val <= 8: age_factor = 1.15
            elif age_val < 2.5: age_factor = 0.8 + (age_val / 5)
            else: age_factor = max(0.6, 1.1 - (age_val - 8) * 0.05)
            base_price *= age_factor
            weight_factor = 1.0 + (min(weight_val, 600) - 300) * 0.001
            base_price *= weight_factor
            if milk_yield_val > 1:
                 milk_factor = 1.0 + (milk_yield_val * 0.05)
                 base_price *= milk_factor
            health_factors = {"Excellent": 1.1, "Good": 1.0, "Fair": 0.85, "Poor": 0.6}
            base_price *= health_factors.get(health_status_val, 0.9)
            if is_pregnant_val: base_price *= 1.1
            estimated_price = max(15000, base_price)

            st.success(translate_text(f"Estimated Valuation Range: ₹ {estimated_price * 0.85:,.0f} - ₹ {estimated_price * 1.15:,.0f}",current_lang))
            ts.caption("Disclaimer: This is an indicative range. Actual market prices vary significantly based on location, pedigree, specific traits, current demand, and negotiation.")

        # --- NEW: Financial Assistance & Loan Calculator Section ---
        st.markdown("---")
        ts.subheader("💰 Financial Assistance & Loan Calculator")

        with st.expander(translate_text("Explore Financial Assistance Options (Information)",current_lang)):
            ts.markdown("""
            Farmers can often avail financial assistance for cattle-related investments through various channels:

            *   **Government Schemes:** Many central and state government schemes offer subsidies, low-interest loans, or grants for purchasing cattle, setting up dairy infrastructure, feed, etc. 
                *   *Refer to the "Govt Schemes" section of this app for details on schemes like KCC, NLM, AHIDF, and state-specific programs.*
            *   **Banks & Financial Institutions:**
                *   **NABARD (National Bank for Agriculture and Rural Development):** Plays a crucial role in refinancing and promoting rural credit. They have various schemes for dairy and animal husbandry.
                *   **Commercial Banks:** Many public and private sector banks offer agricultural loans, including term loans for dairy units, cattle purchase, and working capital.
                *   **Regional Rural Banks (RRBs) & Cooperative Banks:** These institutions are specifically focused on rural credit.
            *   **Microfinance Institutions (MFIs):** Some MFIs provide smaller loans to farmers and livestock rearers.

            **Common Loan Purposes:**
            *   Purchase of milch animals (cows, buffaloes)
            *   Construction of cattle sheds
            *   Purchase of dairy equipment (milking machines, chaff cutters, bulk milk coolers)
            *   Setting up feed units or purchasing fodder
            *   Working capital for day-to-day operations

            **General Requirements (may vary by lender/scheme):**
            *   Land ownership documents (or lease agreements)
            *   Project report or plan for the dairy unit
            *   KYC documents (ID, address proof)
            *   Quotation for cattle/equipment to be purchased
            *   Experience in dairy farming may be preferred
            *   Mortgage/collateral may be required for larger loans

            **It is highly recommended to contact your nearest bank branch, a NABARD office, or your state's animal husbandry department to get the most accurate and up-to-date information on available loan options and eligibility criteria.**
            """)

        st.markdown("---")
        ts.subheader("Simple Loan Repayment Estimator")
        ts.caption("This calculator provides a basic EMI (Equated Monthly Installment) estimate for a loan.")

        col1_loan, col2_loan, col3_loan = st.columns(3)
        with col1_loan:
            loan_amount = st.number_input(translate_text("Loan Amount (Principal) (₹):",current_lang), min_value=1000.0, value=100000.0, step=1000.0, key="loan_amt")
        with col2_loan:
            annual_interest_rate = st.number_input(translate_text("Annual Interest Rate (%):",current_lang), min_value=1.0, max_value=30.0, value=9.0, step=0.1, key="loan_rate")
        with col3_loan:
            loan_tenure_years = st.number_input(translate_text("Loan Tenure (Years):",current_lang), min_value=1, max_value=15, value=5, step=1, key="loan_tenure")
        ts.markdown("**Note:** This is a simplified EMI calculator. Actual loan terms may vary based on lender policies, credit score, and other factors.")

        # Input for milk revenue estimation
        st.markdown("---")
        ts.markdown("**Optional: Estimate Repayment Feasibility based on Milk Revenue**")
        col1_rev, col2_rev, col3_rev = st.columns(3)
        with col1_rev:
            num_milch_animals = st.number_input(translate_text("Number of Milch Animals for Loan Purpose:",current_lang), min_value=0, value=2, step=1, key="loan_cows",)
        with col2_rev:
            avg_daily_milk_per_animal = st.number_input(translate_text("Avg. Daily Milk per Animal (Liters):",current_lang), min_value=0.0, value=8.0, step=0.5, key="loan_milk_yield")    
        with col3_rev:
            milk_price_per_liter = st.number_input(translate_text("Avg. Milk Price per Liter (₹):",current_lang), min_value=10.0, value=35.0, step=1.0, key="loan_milk_price")
        
        other_monthly_income = st.number_input(translate_text("Other Monthly Farm Income (e.g., from dung sale, other crops) (₹):",current_lang), min_value=0.0, value=0.0, step=500.0, key="loan_other_income",)
        monthly_farm_expenses = st.number_input(translate_text("Estimated Monthly Farm Expenses (feed, labor, vet, etc.) (₹):",current_lang), min_value=0.0, value=5000.0, step=500.0, key="loan_expenses")


        if st.button(translate_text("Calculate Loan EMI & Estimate Feasibility",current_lang, key="auto_btn_68"), type="primary", key="btn_loan_calc"):
            if loan_amount > 0 and annual_interest_rate > 0 and loan_tenure_years > 0:
                # Calculate EMI
                monthly_interest_rate = (annual_interest_rate / 100) / 12
                loan_tenure_months = loan_tenure_years * 12

                if monthly_interest_rate > 0:
                    emi = (loan_amount * monthly_interest_rate * (1 + monthly_interest_rate)**loan_tenure_months) / \
                          ((1 + monthly_interest_rate)**loan_tenure_months - 1)
                else: # Handle 0% interest rate case (though unlikely for most loans)
                    emi = loan_amount / loan_tenure_months
                
                total_payment = emi * loan_tenure_months
                total_interest = total_payment - loan_amount

                st.success(translate_text(f"**Estimated Monthly EMI: ₹ {emi:,.2f}**",current_lang))
                st.info(translate_text(f"Total Amount Payable: ₹ {total_payment:,.2f}",current_lang))
                st.info(translate_text(f"Total Interest Payable: ₹ {total_interest:,.2f}",current_lang))

                # Repayment Feasibility based on Milk Revenue
                if num_milch_animals > 0 and avg_daily_milk_per_animal > 0 and milk_price_per_liter > 0:
                    daily_milk_revenue_per_animal = avg_daily_milk_per_animal * milk_price_per_liter
                    total_daily_milk_revenue = num_milch_animals * daily_milk_revenue_per_animal
                    estimated_monthly_milk_revenue = total_daily_milk_revenue * 30 # Assuming 30 days
                    
                    net_monthly_farm_income = (estimated_monthly_milk_revenue + other_monthly_income) - monthly_farm_expenses
                    
                    st.markdown("---")
                    ts.subheader("Repayment Feasibility Estimation:")
                    st.metric(label=translate_text("Estimated Gross Monthly Milk Revenue",current_lang), value=f"₹ {estimated_monthly_milk_revenue:,.2f}")
                    st.metric(label=translate_text("Estimated Net Monthly Farm Income (after expenses)",current_lang), value=f"₹ {net_monthly_farm_income:,.2f}")

                    if net_monthly_farm_income > emi:
                        surplus_after_emi = net_monthly_farm_income - emi
                        st.success(translate_text(f"Feasibility: **Likely Feasible.** Estimated net monthly income (₹{net_monthly_farm_income:,.2f}) is greater than EMI (₹{emi:,.2f}).",current_lang))
                        st.info(translate_text(f"Estimated Monthly Surplus after EMI: ₹ {surplus_after_emi:,.2f}",current_lang))
                    elif net_monthly_farm_income > 0 :
                        st.warning(translate_text(f"Feasibility: **Challenging.** Estimated net monthly income (₹{net_monthly_farm_income:,.2f}) is less than or equal to EMI (₹{emi:,.2f}).",current_lang))
                        ts.markdown("Consider reducing loan amount, extending tenure, or increasing income/reducing expenses.")
                    else:
                        st.error(translate_text(f"Feasibility: **Likely Not Feasible.** Estimated net monthly farm income is zero or negative (₹{net_monthly_farm_income:,.2f}).",current_lang))
                    
                    st.caption(translate_text("This feasibility estimation is indicative and based on averages. Actual income and expenses can vary. Other personal income/expenses are not considered.",current_lang))
                else:
                    st.info(translate_text("Enter milk animal details to estimate repayment feasibility based on milk revenue.",current_lang))

            else:
                st.error(translate_text("Please enter valid loan amount, interest rate, and tenure.",current_lang))

        # --- End of NEW Section ---
        
        # Removed conn.close() as it's handled by @st.cache_resource
        # if conn:
        #     conn.close() 
    else:
        st.error("Database connection failed. Cannot load Price Trends & Calculator.")

# 8. Disease Diagnosis (from original code - no changes needed other than DB context)
elif selected_page == "Diagnosis":
        ts.title("🩺 Cattle Health Diagnosis Hub")
        ts.markdown("Tools and resources for preliminary cattle health assessment. **Always consult a veterinarian for definitive diagnosis and treatment.**")
        st.markdown("---")

        # Removed the third tab for community forum from here
        diag_tab1, diag_tab2 = st.tabs([
            "📝 Symptom-Based Suggester",
            "📸 Image-Based Skin Disease Detector"
        ])

        with diag_tab1:
            # ... (Symptom-Based Suggester logic - no changes) ...
            ts.subheader("🔍 Symptom-Based Disease Suggester (Beta)")
            st.warning(translate_text("**Disclaimer:** This tool provides potential suggestions based on common symptoms. **It is NOT a diagnostic tool.** Always consult a qualified veterinarian.",current_lang), icon="⚠️")
            symptoms_input_diag = st.text_area(translate_text("Enter Observed Symptoms (comma-separated, e.g., fever, cough, loss of appetite):",current_lang), height=100, key="diag_symptoms_input")
            conn = get_connection()
            if conn:
                cursor = conn.cursor()
                if st.button(translate_text("Suggest Potential Diseases (Symptom-Based)", current_lang), type="primary", key="btn_diagnose_symptoms"):
                    if symptoms_input_diag and symptoms_input_diag.strip():
                        symptoms_list_diag = [s.strip().lower() for s in symptoms_input_diag.split(',') if s.strip() and len(s.strip()) > 2]
                        if not symptoms_list_diag:
                            st.warning(translate_text("Please enter valid symptoms (minimum 3 characters each).",current_lang))
                        else:
                            ts.write(f"Searching based on symptoms: **{', '.join(symptoms_list_diag)}**")
                            query_parts_diag = ["LOWER(symptoms) LIKE ?" for _ in symptoms_list_diag]
                            params_diag = [f"%{s}%" for s in symptoms_list_diag]
                            query_diag = f"""
                                SELECT detected_disease, suggested_treatment, severity, symptoms, preventative_measures, notes
                                FROM disease_diagnosis
                                WHERE {' OR '.join(query_parts_diag)}
                                ORDER BY
                                    CASE severity
                                        WHEN 'High' THEN 1
                                        WHEN 'Medium' THEN 2
                                        WHEN 'Low' THEN 3
                                        ELSE 4
                                    END,
                                    RANDOM()
                                LIMIT 7 
                            """
                            try:
                                cursor.execute("SELECT 1 FROM disease_diagnosis LIMIT 1")
                                cursor.execute(query_diag, params_diag)
                                results_diag = cursor.fetchall()
                                if results_diag:
                                    ts.markdown("##### Potential Matches (Based on Symptom Keywords):")
                                    for disease, treatment, severity, db_symptoms, prevention, notes_db in results_diag:
                                        matched_symptoms_display = db_symptoms
                                        for user_symptom in symptoms_list_diag:
                                            import re
                                            pattern = re.compile(re.escape(user_symptom), re.IGNORECASE)
                                            matched_symptoms_display = pattern.sub(f"**{user_symptom.capitalize()}**", matched_symptoms_display)

                                        sev_color = "red" if severity and severity.lower() == 'high' else ("orange" if severity and severity.lower() == 'medium' else "blue")
                                        sev_icon = "🚨" if sev_color == "red" else ("⚠️" if sev_color == "orange" else "ℹ️")

                                        with st.container(border=True):
                                            ts.markdown(f"<font color='{sev_color}'> **{sev_icon} {disease}** (Severity: {severity or 'Unknown'})</font>", unsafe_allow_html=True)
                                            ts.markdown(f"**Matched Symptoms (Keywords):** {matched_symptoms_display}", unsafe_allow_html=True)
                                            ts.markdown(f"**General Recommended Action:** {treatment}")
                                            if prevention: ts.markdown(f"**Preventative Measures:** {prevention}")
                                            if notes_db: st.caption(translate_text(f"Notes: {notes_db}",current_lang))
                                            st.error(translate_text(f"**Critical Reminder:** Consult a veterinarian immediately for proper diagnosis and treatment of {disease}.",current_lang), icon="👩‍⚕️")
                                else:
                                    st.warning(translate_text("No common diseases strongly matched the entered symptoms in the database. Please consult a veterinarian for any health concerns.",current_lang)) 
                            except sqlite3.OperationalError:
                                st.error(translate_text("Database table 'disease_diagnosis' not found or is inaccessible. This feature is unavailable.",current_lang))
                                logger.error("Symptom-based diagnosis: disease_diagnosis table not found.")
                            except sqlite3.Error as e_diag_db:
                                st.error(translate_text(f"Database error during symptom-based lookup: {e_diag_db}",current_lang))
                                logger.error(f"Symptom-based diagnosis DB error: {e_diag_db}")
                            except Exception as e_diag_unexpected:
                                st.error(translate_text(f"An unexpected error occurred: {e_diag_unexpected}",current_lang))
                                logger.error(f"Symptom-based diagnosis unexpected error: {e_diag_unexpected}")
                    else:
                        st.warning(translate_text("Please enter symptoms to get suggestions.",current_lang))
            else:
                st.error(translate_text("Database connection failed. Symptom-based suggester unavailable.",current_lang))
        # Section for Image-Based Skin Disease Detector

        with diag_tab2:
            render_skin_disease_detector_ui()



# 9. Government Schemes (from original code - minor DB context changes)
elif selected_page == "Govt Schemes":
    ts.title("🏛️ Government Schemes Information Hub") # Using ts.title for translation

    # Original English text for descriptions
    desc_text_1_english = "Explore Central and State government schemes relevant to agriculture and animal husbandry."
    desc_text_2_english = "No schemes found for your criteria."
    desc_text_3_english = "No schemes in DB."
    desc_text_4_english = "Database connection failed."
    desc_text_5_english = "Filter by Region:"
    desc_text_6_english = "Filter by Scheme Type:"
    desc_text_7_english = "Search by Scheme Name or Keyword:"
    search_placeholder_english = "e.g., KCC, NLM..."
    found_schemes_subheader_english = "Found {len_schemes} Matching Schemes:"
    official_source_button_english = "🔗 Official Source"
    source_caption_english = "Source: {url}"

    # Use ts.markdown for the main introductory text
    ts.markdown(desc_text_1_english)
    st.markdown("---") # Visual separator, no translation needed

    conn = get_connection() # Assuming get_connection() is defined elsewhere
    if conn:
        cursor = conn.cursor()

        # RE-FETCHING options to ensure correct values for DB query
        available_regions_db_values = ["All India / Central"]
        available_types_db_values = ["All Types"]
        try:
            cursor.execute("SELECT DISTINCT region FROM government_schemes WHERE region IS NOT NULL AND region != '' ORDER BY region ASC")
            available_regions_db_values.extend([r[0] for r in cursor.fetchall()])
            cursor.execute("SELECT DISTINCT type FROM government_schemes WHERE type IS NOT NULL AND type != '' ORDER BY type ASC")
            available_types_db_values.extend([t[0] for t in cursor.fetchall()])
        except sqlite3.Error as e:
             st.error(translate_text(f"Error re-fetching filter options for DB query: {e}", current_lang))

        # Create translated display lists
        display_regions = [translate_text(r, current_lang) for r in available_regions_db_values]
        display_types = [translate_text(t, current_lang) for t in available_types_db_values]

        # Get index of currently selected item (if any) to set selectbox default
        initial_region_idx = display_regions.index(translate_text(available_regions_db_values[0], current_lang))
        initial_type_idx = display_types.index(translate_text(available_types_db_values[0], current_lang))

        col1, col2 = st.columns(2)
        with col1:
            selected_region_display = st.selectbox(
                translate_text(desc_text_5_english, current_lang),
                display_regions,
                index=initial_region_idx
                # Removed key argument from st.selectbox
            )
            selected_region = available_regions_db_values[display_regions.index(selected_region_display)] # Get original DB value

        with col2:
            selected_type_display = st.selectbox(
                translate_text(desc_text_6_english, current_lang),
                display_types,
                index=initial_type_idx
                # Removed key argument from st.selectbox
            )
            selected_type = available_types_db_values[display_types.index(selected_type_display)] # Get original DB value


        # Translate search input label and placeholder
        search_term = st.text_input(
            translate_text(desc_text_7_english, current_lang),
            placeholder=translate_text(search_placeholder_english, current_lang)
            # Removed key argument from st.text_input
        )

        try:
            query = "SELECT name, details, url, region, type FROM government_schemes WHERE 1=1"
            params = []
            if selected_region != "All India / Central":
                query += " AND region = ?"
                params.append(selected_region)
            else: # Explicitly handle "All India / Central" for DB query
                query += " AND (LOWER(region) = ? OR region IS NULL OR region = '' OR LOWER(region) LIKE '%central%' OR LOWER(region) LIKE '%all india%')"
                params.append('all india / central') # Match how it's stored in DB

            if selected_type != "All Types":
                query += " AND type = ?"
                params.append(selected_type)

            if search_term:
                query += " AND (LOWER(name) LIKE ? OR LOWER(details) LIKE ?)"
                params.extend([f"%{search_term.lower()}%"] * 2)

            query += " ORDER BY name ASC"

            cursor.execute("SELECT 1 FROM government_schemes LIMIT 1") # Check table exists
            cursor.execute(query, params)
            schemes = cursor.fetchall()

            st.markdown("---") # Visual separator

            # Translate the subheader and use f-string for length
            ts.subheader(found_schemes_subheader_english.format(len_schemes=len(schemes)))

            if schemes:
                for name, details, url, region_db, type_db in schemes:
                    # Translate name, details, region_db, type_db for display
                    translated_name = translate_text(name, current_lang)
                    translated_details = translate_text(details, current_lang)
                    translated_region_display = translate_text(region_db or 'Central/All India', current_lang)
                    translated_type_display = translate_text(type_db or 'N/A', current_lang)

                    meta_info = [f"📍 {translated_region_display}", f"🏷️ {translated_type_display}"]

                    # Use translated name and meta_info in expander title
                    with st.expander(f"**{translated_name}** {' | '.join(meta_info)}"):
                        # Removed key argument from st.expander
                        st.markdown(translated_details) # Use st.markdown for the details
                        if url and url.strip().startswith("http"):
                            # Translate the link button label
                            st.link_button(translate_text(official_source_button_english, current_lang), url)
                            # Removed key argument from st.link_button
                        elif url:
                            # Translate the caption
                            st.caption(translate_text(source_caption_english.format(url=url.strip()), current_lang))
            elif search_term or selected_region != "All India / Central" or selected_type != "All Types":
                st.info(translate_text(desc_text_2_english, current_lang)) # Translate "No schemes found"
            else:
                st.info(translate_text(desc_text_3_english, current_lang)) # Translate "No schemes in DB"

        except sqlite3.OperationalError:
            st.error(translate_text("DB table 'government_schemes' not found.", current_lang)) # Translate DB error
        except sqlite3.Error as e:
            st.error(translate_text(f"Error fetching schemes: {e}", current_lang)) # Translate general DB error
        except Exception as e:
            st.error(translate_text(f"Unexpected error: {e}", current_lang)) # Translate unexpected error
    else:
        st.error(translate_text(desc_text_4_english, current_lang)) # Translate "Database connection failed"



elif st.session_state.current_page == "Community Network":
    render_main_community_forum_ui()

# 10. Lifecycle Management (from original code, minor path fix)
if selected_page == "Lifecycle Management":
    ts.title("🔄 " + translate_text("Cattle Lifecycle Management Guide", current_lang))
    ts.markdown(translate_text("Essential care and management practices for cattle at different life stages.", current_lang))
    st.markdown("---")

    # --- UPDATED lifecycle_stages DICTIONARY STRUCTURE ---
    lifecycle_stages = {
        "Calf (0-6 months)": {
            "image": "calf.jpeg",
            "focus": translate_text("Immunity, Growth, Weaning", current_lang),
            "general_care": [
                translate_text("**Colostrum:** Feed 10% of body weight within 2-4h of birth.", current_lang),
                translate_text("**Housing:** Clean, dry, warm, draft-free. Individual housing initially.", current_lang),
                translate_text("**Feeding:** Milk replacer/whole milk. Introduce calf starter (18-20% Protein) from day 3-4.", current_lang),
                translate_text("**Water:** Fresh, clean water from day 1.", current_lang),
                translate_text("**Health:** Navel disinfection, monitor scours/pneumonia. Deworming & vaccinations (vet consult).", current_lang),
                translate_text("**Weaning:** Gradual (8-10 weeks), based on starter intake (>1 kg/day).", current_lang)
            ],
            "common_challenges": [
                translate_text("**Calf Scours (Diarrhea):** Leading cause of calf mortality, often due to infections or poor colostrum intake.", current_lang),
                translate_text("**Pneumonia:** Respiratory issues, cough, fever, often triggered by stress or poor ventilation.", current_lang),
                translate_text("**Navel Ill/Joint Ill:** Infections entering through the umbilical cord, leading to lameness or systemic illness.", current_lang),
                translate_text("**Parasites:** Internal and external parasites affecting growth, immunity, and overall health.", current_lang)
            ],
            "key_performance_indicators": [
                translate_text("Daily Weight Gain (DWG): Aim for 0.6-0.8 kg/day for healthy growth.", current_lang),
                translate_text("Weaning Age/Weight: Typically 8-10 weeks or 80-100 kg, indicating successful solid feed intake.", current_lang),
                translate_text("Morbidity/Mortality Rates: Keep low (<5%) to reflect effective health management.", current_lang)
            ]
        },
        "Heifer (6-24 months)": {
            "image": "heif.jpeg",
            "focus": translate_text("Growth, Sexual Maturity, Breeding Prep", current_lang),
            "general_care": [
                translate_text("**Nutrition:** Balanced ration for steady growth (avoid fattening). Target ~60-65% mature BW at first breeding.", current_lang),
                translate_text("**Forage:** Good quality green fodder & hay as primary feed.", current_lang),
                translate_text("**Concentrate:** Supplement as needed (14-16% Protein) based on growth targets.", current_lang),
                translate_text("**Minerals:** Balanced mineral mixture and trace elements are crucial for bone development and fertility.", current_lang),
                translate_text("**Health:** Regular deworming & booster vaccinations. Monitor for external parasites.", current_lang),
                translate_text("**Breeding:** Observe heat (9-15 months). Breed by target weight & age (15-18 months for optimal first calving). Use AI or tested bull.", current_lang)
            ],
            "common_challenges": [
                translate_text("**Poor Growth Rates:** Often due to inadequate nutrition, leading to delayed maturity.", current_lang),
                translate_text("**Reproductive Issues:** Silent heats, anestrus, or ovarian cysts delaying conception.", current_lang),
                translate_text("**Internal Parasites:** Can significantly impede growth and nutrient absorption.", current_lang),
                translate_text("**Lameness:** Due to poor foot health or housing conditions.", current_lang)
            ],
            "key_performance_indicators": [
                translate_text("Age at First Breeding: Target 15-18 months.", current_lang),
                translate_text("Weight at First Breeding: Target 350-400 kg (breed-dependent).", current_lang),
                translate_text("Average Daily Gain (ADG): Monitor to ensure steady development.", current_lang)
            ]
        },
        "Pregnant Cow/Heifer": {
            "image": "preg.jpeg",
            "focus": translate_text("Fetal Growth, Udder Dev, Calving Prep", current_lang),
            "general_care": [
                translate_text("**Early/Mid Gestation (1-6m):** Maintain good body condition, adequate nutrition.", current_lang),
                translate_text("**Late Gestation (7-9m):** Nutrient needs increase significantly (protein, energy, Ca, P). ~25% extra energy, especially in the last trimester.", current_lang),
                translate_text("**Feeding:** High-quality forage + carefully balanced concentrate. Avoid sudden feed changes.", current_lang),
                translate_text("**Minerals:** Crucial balance of Calcium, Phosphorus, Selenium, and Vitamin E for calving and preventing metabolic disorders.", current_lang),
                translate_text("**Health:** Monitor body condition closely. Administer booster vaccinations 4-6 weeks pre-calving.", current_lang),
                translate_text("**Management:** Avoid stress. Prepare a clean, disinfected, and comfortable calving pen 1-2 weeks pre-calving.", current_lang)
            ],
            "common_challenges": [
                translate_text("**Metabolic Diseases:** Milk fever (hypocalcemia), ketosis, often linked to imbalanced nutrition pre-calving.", current_lang),
                translate_text("**Dystocia (Difficult Calving):** Can be caused by large calf, malpresentation, or cow's poor condition.", current_lang),
                translate_text("**Retained Placenta:** Failure to expel fetal membranes post-calving, leading to infections.", current_lang),
                translate_text("**Mastitis:** Increased susceptibility around calving.", current_lang)
            ],
            "key_performance_indicators": [
                translate_text("Body Condition Score (BCS): Maintain 3.5-3.75 at calving for optimal health and lactation.", current_lang),
                translate_text("Calving Ease: Monitor percentage of unassisted calvings.", current_lang),
                translate_text("Calf Viability: Number of live and healthy calves born.", current_lang)
            ]
        },
        "Lactating Cow": {
            "image": "lac.jpeg",
            "focus": translate_text("Milk Production, Health, Re-breeding", current_lang),
            "general_care": [
                translate_text("**Nutrition:** Highest demand! Feed based on milk yield, lactation stage, and body condition. Energy is paramount.", current_lang),
                translate_text("**Energy & Protein:** Key for milk synthesis. Provide high-quality forage supplemented with balanced concentrates (16-18% Protein).", current_lang),
                translate_text("**Water:** Crucial! 4-5 liters of fresh, clean water per liter of milk produced, plus maintenance needs.", current_lang),
                translate_text("**Minerals:** Especially Calcium & Phosphorus. Offer a free-choice mineral mix or incorporate into feed.", current_lang),
                translate_text("**Milking:** Practice hygienic milking (e.g., proper udder cleaning, pre-dipping, post-dipping). Maintain consistent milking times.", current_lang),
                translate_text("**Health:** Continuously monitor for mastitis, lameness, and metabolic diseases (e.g., ketosis).", current_lang),
                translate_text("**Breeding:** Aim to re-breed 60-90 days post-calving to maintain optimal calving interval.", current_lang)
            ],
            "common_challenges": [
                translate_text("**Mastitis:** Inflammation of the udder, common and impacts milk quality and yield.", current_lang),
                translate_text("**Lameness:** Pain and inability to walk normally, affecting feed intake and reproduction.", current_lang),
                translate_text("**Ketosis:** Metabolic disorder due to negative energy balance in early lactation.", current_lang),
                translate_text("**Reproductive Disorders:** Cystic ovaries, metritis, leading to delayed re-conception.", current_lang)
            ],
            "key_performance_indicators": [
                translate_text("Peak Milk Yield: Monitor highest daily milk production.", current_lang),
                translate_text("Days in Milk (DIM) to Conception: Aim for 60-90 days.", current_lang),
                translate_text("Somatic Cell Count (SCC): Indicator of udder health and milk quality.", current_lang),
                translate_text("Milk Fat and Protein Content: Reflects dietary adequacy.", current_lang)
            ]
        },
        "Dry Cow (Non-lactating)": {
            "image": "dry.jpeg",
            "focus": translate_text("Udder Rest, Fetal Growth, Prep for Lactation", current_lang),
            "general_care": [
                translate_text("**Duration:** A critical 45-60 days dry period pre-calving for udder regeneration and fetal development.", current_lang),
                translate_text("**Nutrition:** Lower energy requirements. Maintain body condition (BCS 3.0-3.5). Avoid over-fattening.", current_lang),
                translate_text("**Feeding:** Good quality forage as the primary feed. Low/no concentrate initially, increasing slightly in the last 2-3 weeks.", current_lang),
                translate_text("**Minerals:** Adjust mineral mix (e.g., anionic salts for Calcium management, transition minerals) to prevent metabolic issues post-calving.", current_lang),
                translate_text("**Health:** Ideal time for Dry Cow Therapy (internal antibiotic infusion, consult vet) to prevent mastitis in the next lactation. Monitor overall health closely.", current_lang),
                translate_text("**Management:** Separate from milking herd. Provide comfortable, clean, and well-ventilated housing.", current_lang)
            ],
            "common_challenges": [
                translate_text("**Mastitis (Dry Period):** Infections can occur during the dry period, affecting the next lactation.", current_lang),
                translate_text("**Metabolic Disorders (Pre-Calving):** Hypocalcemia (milk fever) can originate here if diet isn't managed.", current_lang),
                translate_text("**Udder Edema:** Swelling of the udder, often due to high sodium or potassium intake.", current_lang)
            ],
            "key_performance_indicators": [
                translate_text("Dry Period Length: Optimal 45-60 days.", current_lang),
                translate_text("Body Condition Score at Calving: Aim for BCS 3.5.", current_lang),
                translate_text("Incidence of Mastitis in Early Lactation: Reflects dry cow management.", current_lang)
            ]
        },
        "Bull / Breeding Male": {
            "image": "bull.jpeg",
            "focus": translate_text("Libido & Fertility, Soundness, Safe Handling", current_lang),
            "general_care": [
                translate_text("**Nutrition:** Balanced diet for good body condition (avoid both thinness and excessive fatness).", current_lang),
                translate_text("**Feeding:** Good quality forage plus moderate concentrate (12-14% Protein). Adequate minerals (Zinc, Selenium) for reproductive health.", current_lang),
                translate_text("**Exercise:** Provide adequate space for exercise to maintain muscle tone and physical fitness.", current_lang),
                translate_text("**Health:** Regular checks for lameness, reproductive organ health. Annual Breeding Soundness Exam (BSE) is crucial.", current_lang),
                translate_text("**Management:** Handle with extreme caution and use appropriate restraint techniques. Monitor activity levels and libido.", current_lang),
                translate_text("**Biosecurity:** Test for reproductive diseases (e.g., Brucellosis, Vibriosis). Quarantine new arrivals to prevent disease introduction.", current_lang)
            ],
            "common_challenges": [
                translate_text("**Low Libido/Infertility:** Can be caused by poor nutrition, heat stress, disease, or injury.", current_lang),
                translate_text("**Lameness:** Any issue affecting mobility can prevent natural breeding.", current_lang),
                translate_text("**Reproductive Organ Issues:** Infections, injuries, or developmental defects affecting semen quality.", current_lang),
                translate_text("**Aggression:** Bulls can be dangerous; improper handling is a major risk.", current_lang)
            ],
            "key_performance_indicators": [
                translate_text("Semen Quality: Assessed during Breeding Soundness Exam.", current_lang),
                translate_text("Serving Capacity: Number of successful breedings within a set period.", current_lang),
                translate_text("Conception Rate: Percentage of cows settling after breeding by the bull.", current_lang),
                translate_text("Body Condition Score: Maintain BCS 3.0-3.5.", current_lang)
            ]
        }
    }
    # --- END UPDATED lifecycle_stages DICTIONARY STRUCTURE ---

    selected_stage_lc = st.selectbox(
        translate_text("Select Cattle Lifecycle Stage:", current_lang),
        list(lifecycle_stages.keys()),
        key="lc_stage_select"
    )

    if selected_stage_lc:
        stage_info = lifecycle_stages[selected_stage_lc]

        # Use columns for image and main details
        col1, col2 = st.columns([1, 2])

        with col1:
            display_image(stage_info.get("image"), caption=selected_stage_lc) # Display the stage image

        with col2:
            ts.subheader(translate_text("Key Focus:", current_lang) + f" {stage_info['focus']}")

            # General Care Section
            ts.markdown(f"### {translate_text('General Care & Management', current_lang)}")
            for point in stage_info["general_care"]:
                st.markdown(f"- {point}")

            # Common Challenges Section
            if stage_info["common_challenges"]: # Only show if there are challenges
                ts.markdown(f"### {translate_text('Common Challenges & Prevention', current_lang)}")
                for challenge in stage_info["common_challenges"]:
                    st.markdown(f"- {challenge}")

            # Key Performance Indicators Section
            if stage_info["key_performance_indicators"]: # Only show if there are KPIs
                ts.markdown(f"### {translate_text('Key Performance Indicators (KPIs)', current_lang)}")
                for kpi in stage_info["key_performance_indicators"]:
                    st.markdown(f"- {kpi}")

    st.markdown("---")
    st.info(translate_text("This guide provides a general overview. Individual needs may vary, and professional veterinary consultation is always recommended for specific health concerns or detailed management plans.", current_lang), icon="ℹ️")

                
elif st.session_state.current_page == "Fair Price Guide":
    ts.title("📊 Fair Price Guide (Indicative)")
    ts.warning("""
        **Disclaimer:** The price ranges provided here are indicative and for general awareness only.
        They are based on general market estimates or surveys and may not reflect the exact value of a specific
        animal or piece of machinery. Actual transaction prices will vary significantly based on quality,
        age, health, pedigree (for cattle), condition (for machinery), location, seller, buyer, and negotiation.
        Always do your own due diligence.
    """)
    st.markdown("---") # Structural, not translatable

    conn = get_connection()
    if not conn:
        ts.error("Database connection failed.")
        st.stop()
    cursor = conn.cursor()

    guide_type = st.radio(translate_text("Select Guide Type:", current_lang), ("Cattle", "Machinery"), horizontal=True, key="price_guide_type")
    st.markdown("---") # Structural, not translatable

    if guide_type == "Cattle":
        ts.subheader("🐄 Indicative Cattle Price Ranges")
        
        # Filters for Cattle Price Guide
        try:
            cursor.execute("SELECT DISTINCT breed_or_machinery_type FROM indicative_prices WHERE item_type='Cattle' ORDER BY breed_or_machinery_type")
            cattle_breeds_in_guide = [translate_text("All Breeds", current_lang)] + [r[0] for r in cursor.fetchall() if r[0]]
            
            cursor.execute("SELECT DISTINCT category_subtype FROM indicative_prices WHERE item_type='Cattle' ORDER BY category_subtype")
            cattle_categories_in_guide = [translate_text("All Categories", current_lang)] + [r[0] for r in cursor.fetchall() if r[0]]
            
            cursor.execute("SELECT DISTINCT region FROM indicative_prices WHERE item_type='Cattle' ORDER BY region")
            cattle_regions_in_guide = [translate_text("All Regions", current_lang)] + [r[0] for r in cursor.fetchall() if r[0]]

        except sqlite3.Error as e_filter:
            ts.error(f"Error loading filter options: {e_filter}")
            cattle_breeds_in_guide = [translate_text("All Breeds", current_lang)]
            cattle_categories_in_guide = [translate_text("All Categories", current_lang)]
            cattle_regions_in_guide = [translate_text("All Regions", current_lang)]


        fpg_c1, fpg_c2, fpg_c3 = st.columns(3) # Structural, not translatable
        selected_breed_pg = fpg_c1.selectbox(translate_text("Filter by Breed:", current_lang), cattle_breeds_in_guide, key="pg_cattle_breed")
        selected_category_pg = fpg_c2.selectbox(translate_text("Filter by Category:", current_lang), cattle_categories_in_guide, key="pg_cattle_cat")
        selected_region_pg = fpg_c3.selectbox(translate_text("Filter by Region:", current_lang), cattle_regions_in_guide, key="pg_cattle_region")

        query_pg_cattle = "SELECT breed_or_machinery_type, category_subtype, region, price_range_low, price_range_high, notes, data_source, last_updated FROM indicative_prices WHERE item_type='Cattle'"
        params_pg_cattle = []

        if selected_breed_pg != translate_text("All Breeds", current_lang): 
            query_pg_cattle += " AND breed_or_machinery_type = ?"
            params_pg_cattle.append(selected_breed_pg)
        if selected_category_pg != translate_text("All Categories", current_lang): 
            query_pg_cattle += " AND category_subtype = ?"
            params_pg_cattle.append(selected_category_pg)
        if selected_region_pg != translate_text("All Regions", current_lang): 
            query_pg_cattle += " AND region = ?"
            params_pg_cattle.append(selected_region_pg)
        
        query_pg_cattle += " ORDER BY breed_or_machinery_type, category_subtype, region"

        try:
            cursor.execute(query_pg_cattle, params_pg_cattle)
            price_guides_cattle = cursor.fetchall()

            if price_guides_cattle:
                for item in price_guides_cattle:
                    (breed, cat, region, low, high, notes, source, updated) = item
                    with st.container(border=True): # Structural, not translatable
                        ts.markdown(f"**Breed:** {breed} | **Category:** {cat} | **Region:** {region}")
                        # FIX: Use st.metric directly
                        st.metric(label=translate_text("Indicative Price Range", current_lang), value=f"₹{low:,.0f} - ₹{high:,.0f}")
                        if notes: ts.caption(f"Notes: {notes}")
                        if source: ts.caption(f"Source: {source} (Last Updated: {datetime.strptime(updated, '%Y-%m-%d').strftime('%b %Y') if updated else 'N/A'})")
                        st.markdown("---") # Structural, not translatable
            else:
                ts.info(translate_text("No indicative price data found for the selected cattle filters.", current_lang))
        except sqlite3.Error as e_pg_c:
            ts.error(f"Error fetching cattle price guide: {e_pg_c}")


    elif guide_type == "Machinery":
        ts.subheader("🚜 Indicative Machinery Price Ranges")
        try:
            cursor.execute("SELECT DISTINCT type FROM machinery_listings ORDER BY type") # Using type from actual listings
            mach_types_in_guide = [translate_text("All Types", current_lang)] + [r[0] for r in cursor.execute("SELECT DISTINCT breed_or_machinery_type FROM indicative_prices WHERE item_type='Machinery' ORDER BY breed_or_machinery_type").fetchall() if r[0]]
            
            cursor.execute("SELECT DISTINCT category_subtype FROM indicative_prices WHERE item_type='Machinery' ORDER BY category_subtype")
            mach_categories_in_guide = [translate_text("All Categories", current_lang)] + [r[0] for r in cursor.fetchall() if r[0]] # e.g. "Used - Good (35-45HP)"

            cursor.execute("SELECT DISTINCT region FROM indicative_prices WHERE item_type='Machinery' ORDER BY region")
            mach_regions_in_guide = [translate_text("All Regions", current_lang)] + [r[0] for r in cursor.fetchall() if r[0]]

        except sqlite3.Error as e_filter_m:
            ts.error(f"Error loading filter options: {e_filter_m}")
            mach_types_in_guide = [translate_text("All Types", current_lang)]
            mach_categories_in_guide = [translate_text("All Categories", current_lang)]
            mach_regions_in_guide = [translate_text("All Regions", current_lang)]

        fpg_m1, fpg_m2, fpg_m3 = st.columns(3) # Structural, not translatable
        selected_type_pg_m = fpg_m1.selectbox(translate_text("Filter by Machinery Type:", current_lang), mach_types_in_guide, key="pg_mach_type")
        selected_category_pg_m = fpg_m2.selectbox(translate_text("Filter by Category/Condition:", current_lang), mach_categories_in_guide, key="pg_mach_cat")
        selected_region_pg_m = fpg_m3.selectbox(translate_text("Filter by Region:", current_lang), mach_regions_in_guide, key="pg_mach_region")

        query_pg_mach = "SELECT breed_or_machinery_type, category_subtype, region, price_range_low, price_range_high, notes, data_source, last_updated FROM indicative_prices WHERE item_type='Machinery'"
        params_pg_mach = []

        if selected_type_pg_m != translate_text("All Types", current_lang): 
            query_pg_mach += " AND breed_or_machinery_type = ?"
            params_pg_mach.append(selected_type_pg_m)
        if selected_category_pg_m != translate_text("All Categories", current_lang): 
            query_pg_mach += " AND category_subtype = ?"
            params_pg_mach.append(selected_category_pg_m)
        if selected_region_pg_m != translate_text("All Regions", current_lang): 
            query_pg_mach += " AND region = ?"
            params_pg_mach.append(selected_region_pg_m)

        query_pg_mach += " ORDER BY breed_or_machinery_type, category_subtype, region"
        
        try:
            cursor.execute(query_pg_mach, params_pg_mach)
            price_guides_mach = cursor.fetchall()

            if price_guides_mach:
                for item in price_guides_mach:
                    (m_type, cat, region, low, high, notes, source, updated) = item
                    with st.container(border=True): # Structural, not translatable
                        ts.markdown(f"**Type:** {m_type} | **Category/Condition:** {cat} | **Region:** {region}")
                        # FIX: Use st.metric directly
                        st.metric(label=translate_text("Indicative Price Range", current_lang), value=f"₹{low:,.0f} - ₹{high:,.0f}")
                        if notes: ts.caption(f"Notes: {notes}")
                        if source: ts.caption(f"Source: {source} (Last Updated: {datetime.strptime(updated, '%Y-%m-%d').strftime('%b %Y') if updated else 'N/A'})")
                        st.markdown("---") # Structural, not translatable
            else:
                ts.info(translate_text("No indicative price data found for the selected machinery filters.", current_lang))
        except sqlite3.Error as e_pg_m:
            ts.error(f"Error fetching machinery price guide: {e_pg_m}")
    # No conn.close() due to @st.cache_resource
    
    
# Mock payment function for hackathon
def process_mock_payment(amount, donor_name, payment_method, entity_type, entity_id):
    """Simulates a payment processing, for hackathon purposes."""
    st.info(f"Initiating payment of ₹{amount:,.0f} via {payment_method} for {entity_type} ID: {entity_id} by {donor_name}...", icon="💳")
    import time
    time.sleep(2) # Simulate network delay

    # Simulate success/failure randomly or based on some condition
    if amount > 0 and donor_name != "Fail Payment": # A simple condition to simulate failure
        st.success(f"Payment of ₹{amount:,.0f} received successfully! Thank you, {donor_name}!", icon="✅")
        return True
    else:
        st.error("Payment failed. Please try again or choose another method.", icon="❌")
        return False

# --- Main Page Content ---
if st.session_state.current_page == "Donate & Adopt":
    ts.title("💖 " + translate_text("Support Our Mission: Donate & Adopt", current_lang))
    ts.markdown(translate_text("""
    Your generosity can make a significant difference in the lives of indigenous cattle and the farmers who care for them.
    Explore our current campaigns or consider adopting a cow/calf to provide direct, loving support.
    """, current_lang))
    st.markdown("---")

    conn = get_connection()
    if not conn:
        st.error(translate_text("Database connection failed. Please try again later.", current_lang))
        st.stop()
    cursor = conn.cursor()

    tab_campaigns, tab_adopt = st.tabs([translate_text(" IMPACT CAMPAIGNS ", current_lang), translate_text(" ADOPT A COW/CALF ", current_lang)])

    with tab_campaigns:
        st.header("✨ " + translate_text("Impact Campaigns", current_lang))
        st.write(translate_text("Support specific projects making a difference on the ground. Every contribution helps!", current_lang))

        try:
            cursor.execute("""
                SELECT campaign_id, title, description, goal_amount, current_amount,
                       start_date, end_date, image_url, category
                FROM impact_campaigns WHERE status = 'Active' ORDER BY start_date DESC
            """)
            campaigns = cursor.fetchall()

            if not campaigns:
                st.info(translate_text("No active impact campaigns at the moment. Please check back soon!", current_lang))
            else:
                for camp in campaigns:
                    (camp_id, title, desc, goal, current, start, end, img_filename_from_db, category) = camp
                    with st.container(border=True):
                        col_img_camp, col_info_camp = st.columns([1, 2])
                        with col_img_camp:
                            if img_filename_from_db:
                                # Ensure image path is correct, e.g., 'images/campaigns/' + img_filename_from_db
                                display_static_image(img_filename_from_db, caption=translate_text(title, current_lang), use_container_width=True)
                            else:
                                st.image("https://via.placeholder.com/300x200.png?text=" + translate_text("Campaign Image", current_lang), use_container_width=True)

                        with col_info_camp:
                            st.subheader(translate_text(title, current_lang))
                            end_date_formatted = datetime.strptime(end, '%Y-%m-%d').strftime('%d %b %Y') if end else translate_text('Ongoing', current_lang)
                            st.caption(f"{translate_text('Category', current_lang)}: {translate_text(category, current_lang)} | {translate_text('Ends', current_lang)}: {end_date_formatted}")
                            ts.markdown(translate_text(desc, current_lang))

                            progress_percent = 0
                            if goal and goal > 0 and current is not None:
                                progress_percent = min(int((current / goal) * 100), 100)

                            if goal and goal > 0:
                                st.progress(progress_percent / 100, text=f"₹{current:,.0f} {translate_text('raised of', current_lang)} ₹{goal:,.0f} {translate_text('goal', current_lang)} ({progress_percent}%)")
                            elif current is not None:
                                st.info(f"₹{current:,.0f} {translate_text('raised so far for this cause.', current_lang)}")

                            # --- Payment Feature Integration for Campaigns ---
                            st.markdown("---")
                            st.write(translate_text("**Want to contribute?**", current_lang))
                            col_amt, col_name, col_method = st.columns([1, 1, 1])
                            with col_amt:
                                donation_amount = st.number_input(translate_text("Amount (₹)", current_lang), min_value=50.0, step=50.0, value=500.0, key=f"donate_amt_{camp_id}")
                            with col_name:
                                donor_name = st.text_input(translate_text("Your Name (Optional)", current_lang), value=st.session_state.get('username', ''), key=f"donor_name_{camp_id}")
                            with col_method:
                                payment_method = st.selectbox(translate_text("Payment Method", current_lang), ["UPI", "Credit/Debit Card", "Net Banking"], key=f"pay_method_{camp_id}")

                            if st.button(translate_text(f"Donate ₹{donation_amount:,.0f}", current_lang, key="auto_btn_70"), key=f"donate_btn_{camp_id}", type="primary", use_container_width=True):
                                # Call mock payment processor
                                payment_success = process_mock_payment(donation_amount, donor_name if donor_name else "Anonymous", payment_method, "campaign", camp_id)

                                if payment_success:
                                    # Update campaign's current amount in DB
                                    try:
                                        cursor.execute("UPDATE impact_campaigns SET current_amount = current_amount + ? WHERE campaign_id = ?", (donation_amount, camp_id))
                                        conn.commit()
                                        st.success(translate_text(f"Thank you for your generous donation of ₹{donation_amount:,.0f} to '{title}'!", current_lang))

                                        # Log the successful donation
                                        user_id_log = st.session_state.get('user_id') if st.session_state.get('logged_in') else None
                                        cursor.execute("""INSERT INTO donations_log (user_id, campaign_id, adoptable_animal_id, amount, donor_name, payment_method, payment_status, transaction_date)
                                                          VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                                                       (user_id_log, camp_id, None, donation_amount, donor_name if donor_name else "Anonymous", payment_method, 'Completed'))
                                        conn.commit()
                                        st.rerun() # Rerun to update progress bar
                                    except sqlite3.Error as e_update_donate:
                                        st.error(translate_text(f"Error processing donation update: {e_update_donate}", current_lang))
                                        logger.error(f"Error updating campaign/logging donation: {e_update_donate}")
                                else:
                                    st.warning(translate_text("Your donation was not processed. Please try again.", current_lang))

                        st.markdown("---") # Separator between campaigns
        except sqlite3.Error as e_camp:
            st.error(translate_text(f"Error fetching impact campaigns: {e_camp}", current_lang))
            logger.error(f"DB error fetching campaigns: {e_camp}")

    with tab_adopt:
        ts.header("🐄 " + translate_text("Adopt a Cow/Calf", current_lang))
        ts.write(translate_text("Form a special bond and directly support the well-being of an animal in need. You'll receive regular updates and become part of their journey.", current_lang))

        try:
            cursor.execute("""
                SELECT adoptable_animal_id, name, species, breed, sex, approx_age_years, story,
                       health_status, image_url_1, location_info, monthly_sponsorship_cost,
                       gaushala_or_org_name, contact_for_adoption
                FROM adoptable_animals WHERE is_adopted = 0 ORDER BY name
            """) # Only show animals not yet adopted
            adoptable_list = cursor.fetchall()

            if not adoptable_list:
                ts.info(translate_text("All our current animals have found loving adopters! Please check back soon for new animals in need.", current_lang))
            else:
                # Use a responsive grid layout
                num_cols = 2 # Display 2 animals per row for larger screens
                cols_adopt = st.columns(num_cols)

                for i, animal_data in enumerate(adoptable_list):
                    with cols_adopt[i % num_cols]: # Distribute animals across columns
                        (adopt_id, name_a, species_a, breed_a, sex_a, age_a, story_a, health_a,
                         img_filename_from_db, loc_a, cost_a, org_a, contact_a) = animal_data
                        with st.container(border=True):
                            ts.subheader(f"{translate_text(name_a, current_lang)} ({translate_text(species_a, current_lang)})")
                            if img_filename_from_db:
                                display_static_image(img_filename_from_db, caption=translate_text(name_a, current_lang), use_container_width=True)
                            else:
                                st.image("https://via.placeholder.com/300x200.png?text=" + translate_text("Animal Photo", current_lang), use_container_width=True)

                            st.markdown(f"**{translate_text('Age (approx)', current_lang)}:** {age_a or translate_text('N/A', current_lang)} {translate_text('years', current_lang)} | **{translate_text('Sex', current_lang)}:** {translate_text(sex_a, current_lang) or translate_text('N/A', current_lang)}")
                            st.markdown(f"**{translate_text('Breed', current_lang)}:** {translate_text(breed_a, current_lang) or translate_text('N/A', current_lang)}")
                            st.markdown(f"**{translate_text('Location', current_lang)}:** {translate_text(loc_a, current_lang) or translate_text('N/A', current_lang)} ({translate_text('Managed by', current_lang)}: {translate_text(org_a, current_lang) or translate_text('Shelter', current_lang)})")
                            st.markdown(f"**{translate_text('Health', current_lang)}:** {translate_text(health_a, current_lang) or translate_text('N/A', current_lang)}")

                            with st.expander(translate_text(f"Read {name_a}'s Story & Sponsorship Details", current_lang)):
                                st.write(translate_text(story_a, current_lang) or translate_text("No detailed story available yet.", current_lang))
                                st.markdown("---")
                                if cost_a and cost_a > 0:
                                    st.info(f"**{translate_text('Monthly Sponsorship to Support', current_lang)} {translate_text(name_a, current_lang)}:** ₹{cost_a:,.0f}")
                                else:
                                    ts.info(translate_text(f"Support for {name_a} can be discussed with the shelter.", current_lang))

                                # --- Payment/Sponsorship Feature for Adoptions ---
                                if cost_a and cost_a > 0: # Only show sponsorship option if cost is defined
                                    st.markdown("---")
                                    st.write(translate_text(f"**Sponsor {name_a} today!**", current_lang))
                                    col_adopt_name, col_adopt_method = st.columns(2)
                                    with col_adopt_name:
                                        adopter_name = st.text_input(translate_text("Your Name (Optional)", current_lang), value=st.session_state.get('username', ''), key=f"adopter_name_{adopt_id}")
                                    with col_adopt_method:
                                        adopt_payment_method = st.selectbox(translate_text("Payment Method", current_lang), ["UPI", "Credit/Debit Card", "Net Banking"], key=f"adopt_pay_method_{adopt_id}")

                                    if st.button(translate_text(f"Sponsor {name_a} for ₹{cost_a:,.0f}/month", current_lang, key="auto_btn_71"), key=f"sponsor_btn_{adopt_id}", type="primary", use_container_width=True):
                                        payment_success = process_mock_payment(cost_a, adopter_name if adopter_name else "Anonymous", adopt_payment_method, "adoptable_animal", adopt_id)
                                        if payment_success:
                                            # Mark animal as adopted and log sponsorship
                                            try:
                                                # Update animal status (for full adoption)
                                                # If it's just monthly sponsorship, you might log it differently
                                                # For now, let's assume successful sponsorship marks it as 'in progress' or 'adopted'
                                                cursor.execute("UPDATE adoptable_animals SET is_adopted = 1 WHERE adoptable_animal_id = ?", (adopt_id,))
                                                conn.commit()
                                                st.success(translate_text(f"Thank you for sponsoring {name_a}! You will receive updates soon.", current_lang))

                                                user_id_log = st.session_state.get('user_id') if st.session_state.get('logged_in') else None
                                                cursor.execute("""INSERT INTO donations_log (user_id, campaign_id, adoptable_animal_id, amount, donor_name, payment_method, payment_status, transaction_date, is_recurring)
                                                                  VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)""",
                                                               (user_id_log, None, adopt_id, cost_a, adopter_name if adopter_name else "Anonymous", adopt_payment_method, 'Completed', 1)) # is_recurring = 1 for sponsorship
                                                conn.commit()
                                                st.experimental_rerun() # Rerun to remove adopted animal from list
                                            except sqlite3.Error as e_adopt_update:
                                                st.error(translate_text(f"Error processing sponsorship update: {e_adopt_update}", current_lang))
                                                logger.error(f"Error updating adoptable animal/logging sponsorship: {e_adopt_update}")
                                        else:
                                            st.warning(translate_text("Your sponsorship was not processed. Please try again.", current_lang))
                                else:
                                    st.markdown(f"**{translate_text('To inquire about adopting or sponsoring', current_lang)} {translate_text(name_a, current_lang)}, {translate_text('please contact', current_lang)}:**")
                                    st.success(f"{contact_a or translate_text('The respective Gaushala/Organization', current_lang)}")
                                    if st.button(translate_text(f"💖 Inquire about Adopting {name_a}", current_lang, key="auto_btn_72"), key=f"adopt_inq_{adopt_id}", type="primary", use_container_width=True):
                                        st.success(translate_text(f"Thank you for your interest in {name_a}! Please use the contact details above to connect with {translate_text(org_a, current_lang) or translate_text('the shelter', current_lang)}.", current_lang))


                            st.markdown("<br>", unsafe_allow_html=True) # Spacer between animal cards

        except sqlite3.Error as e_adopt:
            st.error(translate_text(f"Error fetching adoptable animals: {e_adopt}", current_lang))
            logger.error(f"DB error fetching adoptable animals: {e_adopt}")

    # No conn.close() here due to @st.cache_resource

# --- Ensure this is the last `elif` or followed by an `else` for unknown pages ---
# else:
#    if selected_page not in ["Login", "Register", "Logout", "Home", ...all your public pages...]: # Avoid error for standard pages
#        st.error("Page not found or you do not have access.")
#        logger.warning(f"Attempt to access unknown/restricted page: {selected_page} by user {st.session_state.get('username', 'Guest')}")

# --- Footer ---
st.markdown("---")
st.caption("Kamadhenu Program App v1.2 © 2024-2025 | Empowering Sustainable Indian Farming") # Updated version
