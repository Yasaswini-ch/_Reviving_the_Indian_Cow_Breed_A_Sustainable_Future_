import streamlit as st
from streamlit_option_menu import option_menu
from googletrans import Translator, LANGUAGES
import requests # Keep requests as it was in the original file
import pandas as pd
import google.generativeai as genai
import os
import base64
import io
import random
from dotenv import load_dotenv
from PIL import Image
import sqlite3

# Imports specifically needed for the integrated Roboflow logic
import cv2
import numpy as np
from roboflow import Roboflow # Requires 'pip install roboflow'
import supervision as sv # Requires 'pip install supervision'
import uuid # Standard library
import traceback # Standard library
import logging # Standard library

# --- Configuration ---
st.set_page_config(
    page_title="Kamadhenu Program",
    page_icon="🐄",
    layout="wide" # Use wide layout
)

# --- Initialize Logging (Optional but Recommended) ---
# You can comment this out if you prefer no logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables & API Keys ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY") # <<< ADDED: Load Roboflow key
# Make sure this BACKEND_URL is correct for your deployment (e.g., localhost or deployed URL)
# BACKEND_URL variable is now only relevant if other parts of the app used it,
# but the "Identify Breed" section will not use it anymore.
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/predict/") # Kept as per original file
# --- Load Environment Variables & API Keys ---

# --- Roboflow Configuration (Added for integrated logic) ---
ROBOFLOW_PROJECT_ID = "cattle-breed-9rfl6-xqimv-mqao3" # Verify this is correct
ROBOFLOW_MODEL_VERSION = 6
CONFIDENCE_THRESHOLD = 40
OVERLAP_THRESHOLD = 30


# --- Initialize Google Generative AI API ---
# (Keep your original Gemini initialization logic)
gemini_model = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        try:
            # <<< KEPT YOUR ORIGINAL MODEL NAME >>>
            gemini_model = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("Google AI Model (gemini-2.0-flash) initialized.") # Log success
        except Exception as model_err:
             # <<< KEPT YOUR ORIGINAL WARNING MESSAGE REFERENCE >>>
             st.warning(f"Could not initialize Google AI Model (gemini-1.5-pro-latest): {model_err}. Chatbot might not function.", icon="⚠️")
             logger.warning(f"Google AI Model initialization failed: {model_err}")
             gemini_model = None
    except Exception as e:
        st.error(f"Error configuring Google AI SDK: {e}")
        logger.error(f"Google AI SDK Config Error: {e}\n{traceback.format_exc()}")
        GOOGLE_API_KEY = None
else:
    if os.path.exists(".env"):
         st.warning("Google API key not found in the .env file! Chatbot requires a valid GOOGLE_API_KEY.", icon="🔑")
    else:
         st.warning(".env file not found. Chatbot functionality requires a .env file with a valid GOOGLE_API_KEY.", icon="📄")


# --- Initialize Roboflow Model (Cached) ---
# <<< ADDED: Function to load and cache the Roboflow model >>>
@st.cache_resource
def load_roboflow_model():
    """Loads the Roboflow model, returns None on failure."""
    if not ROBOFLOW_API_KEY:
        st.error("Roboflow API Key (ROBOFLOW_API_KEY) not found in environment variables. Breed identification disabled.", icon="🔑")
        return None
    try:
        logger.info("Initializing Roboflow (cached)...")
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        logger.info(f"Accessing workspace (cached)...")
        workspace = rf.workspace()
        logger.info(f"Accessing project: {ROBOFLOW_PROJECT_ID} (cached)")
        project = workspace.project(ROBOFLOW_PROJECT_ID)
        logger.info(f"Loading model version: {ROBOFLOW_MODEL_VERSION} (cached)")
        model = project.version(ROBOFLOW_MODEL_VERSION).model
        logger.info("Roboflow model loaded successfully (cached).")
        return model
    except Exception as e:
        st.error(f"Failed to initialize Roboflow model: {e}. Check API key, project ID, version, and network connection. Breed identification disabled.")
        logger.error(f"Roboflow Initialization Error: {e}\n{traceback.format_exc()}")
        return None

# <<< ADDED: Load the model when the script runs >>>
roboflow_model = load_roboflow_model()


# --- Database Connection ---
@st.cache_resource # Cache the connection for efficiency
def get_connection():
    """Establishes connection to the SQLite database."""
    try:
        db_name = 'Cows.db'
        logger.info(f"Connecting to database: {db_name}") # Log which DB is used
        return sqlite3.connect(db_name, check_same_thread=False)
    except sqlite3.Error as e:
        st.error(f"Database connection error: {e}")
        logger.error(f"Database Connection Error: {e}\n{traceback.format_exc()}")
        return None

# --- Helper Functions ---
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


def display_image(image_path, caption="", width=None, use_container_width=True):
    """Safely displays an image if it exists using st.image."""
    img = load_image(image_path)
    if img:
        st.image(img, caption=caption, use_container_width=use_container_width if width is None else False, width=width)

    elif image_path:
        # Use logger instead of st.warning to avoid cluttering UI for optional images
        logger.warning(f"display_image: Image not found: {os.path.basename(image_path)}")
        st.warning(f"Image not found: {os.path.basename(image_path)}", icon="🖼️")



# --- Cattle Breed Data (Moved here for better access if needed elsewhere) ---
CATTLE_BREEDS_DATA = [
    {"name": "Gir", "region": "Gujarat", "milk_yield": 12, "strength": "High", "lifespan": 18, "image": "images/gir.jpg"},
    {"name": "Sahiwal", "region": "Punjab", "milk_yield": 14, "strength": "Medium", "lifespan": 20, "image": "images/sahiwal.jpg"},
    {"name": "Ongole", "region": "Andhra Pradesh", "milk_yield": 10, "strength": "Very High", "lifespan": 22, "image": "images/ongole.jpg"},
    {"name": "Punganur", "region": "Andhra Pradesh", "milk_yield": 6, "strength": "Low", "lifespan": 15, "image": "images/punganur.jpg"},
    {"name": "Amrit Mahal", "region": "Karnataka", "milk_yield": 7, "strength": "High", "lifespan": 18, "image": "images/amritmahal.jpg"},
    {"name": "Deoni", "region": "Maharashtra", "milk_yield": 9, "strength": "Medium", "lifespan": 19, "image": "images/deoni.jpeg"},
    {"name": "Hallikar", "region": "Karnataka", "milk_yield": 8, "strength": "Very High", "lifespan": 20, "image": "images/hallikar.jpg"},
    {"name": "Kankrej", "region": "Gujarat", "milk_yield": 11, "strength": "High", "lifespan": 21, "image": "images/kankrej.jpg"},
    {"name": "Krishna Valley", "region": "Karnataka", "milk_yield": 7, "strength": "Very High", "lifespan": 19, "image": "images/krishna_valley.jpg"},
    {"name": "Malnad Gidda", "region": "Karnataka", "milk_yield": 5, "strength": "Medium", "lifespan": 16, "image": "images/malnad_gidda.jpeg"},
    {"name": "Rathi", "region": "Rajasthan", "milk_yield": 10, "strength": "Medium", "lifespan": 20, "image": "images/rathi.jpg"},
    {"name": "Red Sindhi", "region": "Sindh (Origin)", "milk_yield": 11, "strength": "High", "lifespan": 22, "image": "images/red_sindhi.jpg"}, # Adjusted region for clarity
    {"name": "Tharparkar", "region": "Rajasthan", "milk_yield": 9, "strength": "Medium", "lifespan": 21, "image": "images/tharparkar.jpg"}
]

selected_page = option_menu(
    menu_title=None,  # No title needed
    options=[
        "Home", "Breed Info", "Breeding", "Eco Practices",
        "Identify Breed", "Chatbot", "Price Trends", "Diagnosis",
        "Govt Schemes", "Lifecycle Management"
    ],
    icons=[
        "house-gear-fill",  # Home
        "info-square-fill",  # Breed Info
        "heart-pulse-fill",  # Breeding
        "recycle",  # Eco Practices
        "camera-fill",  # Identify Breed
        "chat-quote-fill",  # Chatbot
        "graph-up-arrow",  # Price Trends
        "clipboard2-pulse-fill",  # Diagnosis
        "bank",  # Govt Schemes
        "arrow-repeat"  # Lifecycle Mgmt
    ],
    menu_icon="cow",  # Changed icon
    default_index=0,  # Start on the Home page
    orientation="horizontal",
    styles={
        "container": {
            "padding": "5px",  # Reduced padding for a smaller overall size
            "background-color": "#e8f5e9",  # Lighter green background
            "border-radius": "6px",  # Slightly rounded edges
        },
        "icon": {
            "color": "#1e8449",  # Darker green icons
            "font-size": "16px"  # Reduced icon size
        },
        "nav-link": {
            "font-size": "12px",  # Reduced text size for non-selected items
            "font-weight": "500",  # Medium boldness
            "color": "#000000",  # Black text
            "text-align": "center",
            "margin": "0px 5px",  # Reduced space between links
            "--hover-color": "#c8e6c9",  # Light green hover background
            "padding": "6px 8px"  # Reduced padding for clickable area
        },
        "nav-link-selected": {
            "background-color": "#2e7d32",  # Darker green background for selected
            "color": "#ffffff",  # White text for selected item
            "font-weight": "600",  # Bold text for selected item
        },
    }
)
# --- Page Content ---

# 1. Home Page
if selected_page == "Home":
    st.title("🐄 Kamadhenu Program: Sustainable Futures for Indian Farming 🇮🇳")
    st.markdown("---")

    # Banner/Hero Section
    col_img, col_txt = st.columns([1, 2])
    with col_img:
        display_image("images/home1.jpeg", use_container_width=True) # Make sure you have this image
    with col_txt:
        st.subheader("Empowering Farmers, Conserving Heritage, Building Resilience")
        st.markdown("""
            Welcome to the digital hub for the Kamadhenu Program, dedicated to revitalizing Indian agriculture through:
            *   **Conservation:** Protecting and promoting valuable indigenous cattle breeds.
            *   **Innovation:** Utilizing AI and data for smarter farming decisions.
            *   **Sustainability:** Championing eco-friendly practices for long-term prosperity.
            *   **Knowledge:** Providing accessible information and tools for farmers.

            *Navigate using the bar above to explore features like breed identification, sustainable practice guides, market insights, and more.*
        """)
        st.link_button("Learn About Indigenous Breeds", "https://en.wikipedia.org/wiki/Indigenous_cattle_breeds_of_India")


    st.markdown("---")
    st.header("Key Features at a Glance")
    col1, col2, col3= st.columns(3)
    with col1:
        st.info("**Breed Identification & Info**", icon="🧬")
        st.caption("Use AI to identify breeds from images. Access detailed info on indigenous cattle.")
    with col2:
        st.info("**Eco-Friendly Practices Guide**", icon="🌱")
        st.caption("Learn about sustainable techniques like organic farming, water conservation, and waste management.")
    with col3:
        st.info("**Health & Lifecycle Management**", icon="❤️‍🩹")
        st.caption("Get tips for disease diagnosis assistance and managing cattle through different life stages.")
    #with col4:
        #st.subheader("Project Updates")
        #st.success("Launched AI Breed Identification Beta!")
        #st.success("Added new Government Schemes data for Gujarat.")


# 2. Cattle Breed Information
elif selected_page == "Breed Info":
    st.title("🐄 Indigenous Indian Cattle Breed Profiles")
    st.markdown("Discover the unique characteristics, origins, and utility of India's native cattle breeds.")
    st.markdown("---")

    # Use the globally defined data
    cattle_breeds = CATTLE_BREEDS_DATA

    # --- Filters ---
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        search_query = st.text_input("🔍 Search by Breed Name:", placeholder="E.g., Sahiwal, Gir...")
    with col2:
        unique_regions = sorted(list(set(breed["region"] for breed in cattle_breeds)))
        selected_region = st.selectbox("🌍 Filter by Region:", ["All"] + unique_regions)
    with col3:
        sort_options = ["Name", "Milk Yield", "Strength", "Lifespan"]
        sort_option = st.selectbox("📊 Sort by:", sort_options)

    # --- Filtering Logic ---
    filtered_breeds = cattle_breeds
    if search_query:
        filtered_breeds = [b for b in filtered_breeds if search_query.lower() in b["name"].lower()]
    if selected_region != "All":
        filtered_breeds = [b for b in filtered_breeds if b["region"] == selected_region]

    # --- Sorting Logic ---
    if sort_option == "Milk Yield":
        filtered_breeds = sorted(filtered_breeds, key=lambda x: x["milk_yield"], reverse=True)
    elif sort_option == "Lifespan":
        filtered_breeds = sorted(filtered_breeds, key=lambda x: x["lifespan"], reverse=True)
    elif sort_option == "Strength":
        strength_order = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
        # Handle potential missing keys gracefully with .get()
        filtered_breeds = sorted(filtered_breeds, key=lambda x: strength_order.get(x.get("strength", "Low"), 1), reverse=True)
    else: # Sort by Name (default)
        filtered_breeds = sorted(filtered_breeds, key=lambda x: x["name"])

    # --- Display Breeds ---
    if filtered_breeds:
        cols = st.columns(3) # Display 3 breeds per row
        for i, breed in enumerate(filtered_breeds):
            with cols[i % 3]:
                with st.container(border=True):
                    st.subheader(f"{breed['name']}")
                    display_image(breed.get("image", ""), caption=f"{breed['name']} ({breed['region']})")
                    st.markdown(f"**🥛 Avg. Milk Yield:** {breed['milk_yield']} L/day")
                    st.markdown(f"**💪 Strength/Draft:** {breed['strength']}")
                    st.markdown(f"**⏳ Avg. Lifespan:** {breed['lifespan']} years")
                    # Add more details if available, e.g., special characteristics
                    # st.caption(f"Known for: {breed.get('special_trait', 'N/A')}")
    else:
        st.warning("⚠️ No breeds match your current filters.")

# 3. Breeding Program
elif selected_page == "Breeding":
    st.title("🧬 Breeding Program Manager")
    st.markdown("Plan, suggest, and track cattle breeding activities for desired traits.")
    st.markdown("---")

    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        # Ensure tables exist (simple check)
        try:
            cursor.execute("SELECT 1 FROM breeding_pairs LIMIT 1;")
            cursor.execute("SELECT 1 FROM offspring_data LIMIT 1;")
        except sqlite3.OperationalError:
             st.error("Database tables (breeding_pairs, offspring_data) not found. Please initialize the database correctly.")
             conn = None # Prevent further operations

    # Check conn again in case it was set to None above
    if conn:
        cursor = conn.cursor() # Create cursor here, after checks
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Suggest New Pairing")
            cattle_1 = st.text_input("Enter Name/ID of Cattle 1:")
            cattle_2 = st.text_input("Enter Name/ID of Cattle 2:")
            goal = st.selectbox("Select Primary Breeding Goal", ["High Milk Yield", "Disease Resistance", "Drought Tolerance", "Breed Purity", "Temperament", "Dual Purpose (Milk & Draft)"])

            if st.button("Suggest Pair", type="primary"):
                if cattle_1 and cattle_2 and cattle_1.strip().lower() != cattle_2.strip().lower():
                    try:
                        # Placeholder: In reality, fetch traits from DB based on cattle_1, cattle_2
                        genetic_score = random.randint(55, 95) # Example score
                        status = "Recommended" if genetic_score > 75 else "Consider" if genetic_score > 60 else "Evaluate Carefully"
                        notes = f"Goal: {goal}. Est. Compatibility: {genetic_score}%. "
                        if genetic_score < 65: notes += "Potential mismatch in some traits, verify records."

                        # Make sure cursor is defined before using it
                        cursor.execute("""
                            INSERT INTO breeding_pairs (cattle_1, cattle_2, goal, genetic_score, status, notes)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (cattle_1.strip(), cattle_2.strip(), goal, genetic_score, status, notes))
                        #conn.commit() # Commit the transaction
                        st.success(f"Pairing suggestion logged for {cattle_1} & {cattle_2}.")
                        st.info(f"Goal: {goal}, Score: {genetic_score}%, Status: {status}")

                    except sqlite3.Error as e:
                        st.error(f"Database error saving suggestion: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                else:
                    st.error("Please enter two different, non-empty cattle names/IDs.")

        with col2:
            st.subheader("Recent Breeding Records")
            tab1, tab2 = st.tabs(["📈 Suggestions Log", "🍼 Offspring Records"])

            with tab1:
                try:
                    cursor.execute("""
                        SELECT cattle_1, cattle_2, goal, genetic_score, status, timestamp
                        FROM breeding_pairs ORDER BY timestamp DESC LIMIT 10
                        """)
                    pairs = cursor.fetchall()
                    if pairs:
                        df_pairs = pd.DataFrame(pairs, columns=["Cattle 1", "Cattle 2", "Goal", "Score", "Status", "Timestamp"])
                        df_pairs['Timestamp'] = pd.to_datetime(df_pairs['Timestamp'], errors='coerce', format='ISO8601', cache=True)
                        df_pairs['Timestamp'] = df_pairs['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                        st.dataframe(df_pairs, use_container_width=True, hide_index=True)
                    else:
                        st.info("No breeding suggestions recorded yet.")
                except sqlite3.Error as e:
                    st.warning(f"Could not fetch breeding suggestions: {e}")

            with tab2:
                try:
                    cursor.execute("""
                        SELECT parent_1, parent_2, offspring_id, breed, dob, sex
                        FROM offspring_data ORDER BY timestamp DESC LIMIT 10
                        """)
                    offspring = cursor.fetchall()
                    if offspring:
                        df_offspring = pd.DataFrame(offspring, columns=["Parent 1", "Parent 2", "Offspring ID", "Breed", "DOB", "Sex"])
                        st.dataframe(df_offspring, use_container_width=True, hide_index=True)
                    else:
                        st.info("No offspring records yet.")
                except sqlite3.Error as e:
                     st.info(f"Offspring data could not be retrieved or table is empty. Error: {e}")

        # conn.close() # <-- REMOVED THIS LINE

    # No 'else' needed here as error is shown if conn is None


# 4. Eco-Friendly Practices (Expanded Content)
elif selected_page == "Eco Practices":
    st.title("🌱 Eco-Friendly & Sustainable Farming Practices")
    st.markdown("""
        Adopt practices that benefit the environment, improve soil health, conserve resources, and enhance long-term farm resilience.
    """)
    st.markdown("---")

    practices = {
        "Organic Farming": {
            "icon": "🌿",
            "description": "Avoids synthetic fertilizers, pesticides, GMOs. Relies on natural inputs and processes.",
            "benefits": [
                "Improves soil health and biodiversity.",
                "Reduces water pollution from chemical runoff.",
                "Produces potentially healthier food (residue-free).",
                "Can fetch premium prices for certified produce."
            ],
            "implementation": [
                "Use compost, manure, green manures for fertility.",
                "Employ crop rotation, companion planting, biological pest control.",
                "Source organic seeds/inputs.",
                "Maintain buffer zones from conventional farms.",
                "Certification process required for 'Organic' label (can be complex/costly)."
            ],
            "challenges": ["Potentially lower yields initially", "Higher labor input", "Weed and pest control can be difficult."]
        },
        "Crop Rotation": {
            "icon": "🔄",
            "description": "Systematically changing the type of crop grown on a piece of land each season or year.",
            "benefits": [
                "Improves soil structure and fertility (e.g., legumes fix nitrogen).",
                "Breaks pest and disease cycles specific to certain crops.",
                "Suppresses weeds by alternating competitive crops.",
                "Distributes nutrient uptake from different soil depths."
            ],
            "implementation": [
                "Plan rotation sequences considering crop families (avoid planting related crops consecutively).",
                "Include deep-rooted and shallow-rooted crops.",
                "Incorporate legume cover crops.",
                "Consider market demand and crop suitability.",
            ],
             "challenges": ["Requires careful planning", "Market fluctuations for different crops."]
        },
        "Water Conservation": {
            "icon": "💧",
            "description": "Using water resources efficiently in agriculture.",
            "benefits": [
                "Saves a critical resource, especially in water-scarce regions.",
                "Reduces energy costs for pumping.",
                "Minimizes soil erosion and nutrient leaching.",
                "Can improve crop yields by providing water directly to roots."
            ],
            "implementation": [
                "**Drip Irrigation:** Delivers water directly to the root zone.",
                "**Sprinkler Systems:** More efficient than flood irrigation.",
                "**Rainwater Harvesting:** Collect and store rainwater in ponds or tanks.",
                "**Mulching:** Covering soil (organic or plastic) reduces evaporation.",
                "**Laser Land Leveling:** Creates uniform slope for efficient surface irrigation.",
                "**Contour Farming/Bunds:** Slows water runoff on slopes."
            ],
             "challenges": ["Initial investment cost for systems like drip irrigation", "Requires maintenance."]
        },
        "Integrated Pest Management (IPM)": {
            "icon": "🐞",
            "description": "Holistic approach using multiple tactics to control pests, minimizing reliance on chemical pesticides.",
            "benefits": [
                "Reduces pesticide use and environmental contamination.",
                "Protects beneficial insects (pollinators, predators).",
                "Lowers risk of pesticide resistance.",
                "Can be more cost-effective long-term."
            ],
            "implementation": [
                "**Monitoring:** Regularly scout fields to identify pests and assess damage levels.",
                "**Cultural Controls:** Crop rotation, resistant varieties, sanitation.",
                "**Biological Controls:** Introduce or encourage natural enemies (predators, parasitoids).",
                "**Physical/Mechanical Controls:** Traps, barriers, hand-picking.",
                "**Chemical Controls:** Use targeted, least-toxic pesticides only when necessary (based on thresholds)."
            ],
             "challenges": ["Requires knowledge of pest lifecycles and natural enemies", "Can be more complex than simple spraying."]
        },
         "Manure Management": {
            "icon": "💩",
            "description": "Proper handling, storage, and application of animal manure to utilize nutrients and prevent pollution.",
            "benefits": [
                "Recycles valuable nutrients (N, P, K) back to the soil.",
                "Improves soil organic matter and structure.",
                "Reduces reliance on synthetic fertilizers.",
                "Prevents water contamination from runoff.",
                "Potential for biogas generation."
            ],
            "implementation": [
                "**Collection:** Regular collection from sheds/pens.",
                "**Storage:** Covered storage (pits or heaps) to prevent nutrient loss and runoff.",
                "**Composting:** Speeds up decomposition, reduces pathogens, stabilizes nutrients.",
                "**Application:** Apply based on soil tests and crop needs, incorporate into soil quickly.",
                "Avoid applying near water bodies or during heavy rain."
            ],
             "challenges": ["Requires labor and space for handling/storage", "Odor management", "Pathogen risks if not composted properly."]
        },
        "Vermicomposting": {
             "icon": "🪱",
             "description": "Using earthworms (like Eisenia fetida) to decompose organic waste into high-quality compost (vermicast).",
             "benefits": [
                 "Produces nutrient-rich organic fertilizer quickly.",
                 "Improves soil aeration, water retention, and microbial activity.",
                 "Diverts organic waste from landfills/burning.",
                 "Vermicast can suppress some soil-borne diseases."
             ],
             "implementation": [
                 "Use suitable bins or pits with drainage.",
                 "Maintain optimal moisture (~70%) and temperature (15-25°C).",
                 "Feed worms appropriate organic matter (cow dung, crop residues, kitchen waste - avoid oily/meat).",
                 "Harvest vermicast periodically.",
             ],
             "challenges": ["Requires specific worm species", "Sensitive to temperature and moisture extremes", "Needs regular management."]
        },
         "Biogas Production": {
             "icon": "🔥",
             "description": "Anaerobic digestion of organic matter (mainly cow dung) to produce methane gas for fuel and nutrient-rich slurry.",
             "benefits": [
                 "Provides clean, renewable cooking fuel, reducing reliance on firewood/LPG.",
                 "Produces high-quality organic fertilizer (slurry).",
                 "Improves sanitation by managing waste.",
                 "Reduces greenhouse gas emissions (methane capture)."
             ],
             "implementation": [
                 "Construct a biogas digester (fixed dome or floating drum type).",
                 "Feed daily with a mixture of dung and water.",
                 "Use the produced gas for cooking/lighting via pipes.",
                 "Utilize the slurry as fertilizer after storage.",
             ],
             "challenges": ["Initial construction cost", "Requires consistent supply of dung/water", "Temperature affects gas production."]
        },
        "Agroforestry": {
            "icon": "🌳",
            "description": "Integrating trees and shrubs with crops and/or livestock on the same land.",
            "benefits": [
                "Diversifies farm income (timber, fruit, fodder).",
                "Improves soil health (nutrient cycling, erosion control).",
                "Enhances biodiversity (habitat for birds, insects).",
                "Provides shade for livestock, acts as windbreak.",
                "Sequester carbon."
            ],
            "implementation": [
                "Choose suitable tree species compatible with crops/livestock.",
                "Design spatial arrangement (alley cropping, boundary planting, silvopasture).",
                "Manage trees (pruning, thinning) to minimize competition with crops.",
            ],
             "challenges": ["Competition for light, water, nutrients between trees and crops", "Longer time frame for returns from trees."]
        },

        "Rotational Grazing": {
            "icon": "🌱",
            "description": "A livestock management strategy that involves dividing pasture into sections and rotating grazing areas to optimize grass growth and soil health.",
            "benefits": [
                "Prevents overgrazing and allows vegetation to recover.",
                "Improves soil fertility by evenly distributing manure.",
                "Enhances pasture biodiversity and forage quality.",
                "Reduces erosion and maintains healthy ground cover."
            ],
            "implementation": [
                "Divide pasture into multiple paddocks or sections.",
                "Rotate livestock between paddocks based on grass growth and recovery rates.",
                "Provide access to clean water in each grazing area.",
                "Monitor pasture health regularly to adjust grazing schedules."
                ],
                "challenges": [
                    "Initial setup can be resource-intensive (fences, water systems).",
                    "Requires regular monitoring and management of livestock.",
                    "May need supplemental feed during pasture recovery periods."
                ]
            }
    }

    # Create tabs dynamically
    practice_names = list(practices.keys())
    tabs = st.tabs([f"{practices[name]['icon']} {name}" for name in practice_names])

    for i, name in enumerate(practice_names):
        with tabs[i]:
            practice = practices[name]
            st.subheader(f"{practice['icon']} {name}")
            st.markdown(f"**Description:** {practice['description']}")

            st.markdown("**Key Benefits:**")
            for benefit in practice['benefits']:
                st.markdown(f"- {benefit}")

            st.markdown("**Basic Implementation Steps:**")
            for step in practice['implementation']:
                st.markdown(f"- {step}")

            if 'challenges' in practice:
                 st.markdown("**Potential Challenges:**")
                 for challenge in practice['challenges']:
                     st.markdown(f"- {challenge}")

    st.markdown("---")
    st.header("🛠️ Tools for Sustainability Assessment")
    # Keep the calculators as expanders
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🌍 Carbon Footprint Estimator"):
            # ... (calculator code remains the same) ...
            st.markdown("Estimate your farm's approximate **monthly** carbon emissions.")
            fuel_usage = st.number_input("Diesel/Petrol Usage (Liters/month):", min_value=0.0, step=10.0, key="fuel_c")
            fertilizer_usage = st.number_input("Nitrogen Fertilizer Usage (Kg N/month):", min_value=0.0, step=5.0, key="fert_c") # Be specific about N content
            livestock_count = st.number_input("Number of Adult Cattle:", min_value=0, step=1, key="cows_c")
            rice_paddy_area = st.number_input("Area under Rice Paddy (Acres, if applicable):", min_value=0.0, step=0.1, key="rice_c")

            if st.button("Estimate Footprint", key="btn_carbon"):
                # Refined simplified factors (still illustrative)
                fuel_emission = fuel_usage * 2.68 # kg CO2e per liter diesel
                fertilizer_emission = fertilizer_usage * 4.5 # kg CO2e per kg N (includes production+application N2O estimate)
                livestock_emission = livestock_count * (1800 / 12) # kg CO2e per head per year (enteric fermentation) / 12 months
                rice_emission = rice_paddy_area * 50 # kg CO2e per acre per month (highly variable methane estimate)
                total_emissions = fuel_emission + fertilizer_emission + livestock_emission + rice_emission
                st.success(f"Estimated Monthly Footprint: ~{total_emissions:.1f} kg CO₂e")
                st.caption("Note: This is a rough estimate based on general factors.")

    with col2:
        with st.expander("💧 Water Usage Calculator"):
             # ... (calculator code remains the same) ...
            st.markdown("Estimate monthly water usage for irrigation.")
            field_size = st.number_input("Irrigated Field Size (Acres):", min_value=0.0, step=0.5, key="field_w")
            irrigation_per_acre = st.number_input("Avg. Daily Irrigation Depth per Acre (mm):", min_value=0.0, step=1.0, value=5.0, key="depth_w") # Use mm depth
            days_irrigated = st.slider("Number of Irrigation Days per Month:", 1, 31, 20, key="days_w")

            if st.button("Estimate Water Usage", key="btn_water"):
                 # 1 acre = 4046.86 sq meters. 1 mm depth = 0.001 meters. Volume = Area * Depth. 1 cubic meter = 1000 Liters.
                 liters_per_acre_per_day = 4046.86 * (irrigation_per_acre / 1000) * 1000
                 monthly_water_usage = field_size * liters_per_acre_per_day * days_irrigated
                 st.success(f"Estimated Monthly Water Usage: {monthly_water_usage:,.0f} Liters")
                 st.caption(f"(Based on {irrigation_per_acre} mm/day application)")


elif selected_page == "Identify Breed":
    st.title("📸 AI Cattle Breed Identification")
    st.markdown("Upload a clear image of a cow for AI identification.")
    st.markdown("---")

    # Check if Roboflow model loaded successfully
    if not roboflow_model:
        st.error("Roboflow model failed to load. Breed Identification unavailable.", icon="🚫")
    else:
        # Use key from original code
        uploaded_file = st.file_uploader("Choose an image (JPG, PNG)...", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

        if uploaded_file:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Uploaded Image")
                st.image(uploaded_file, use_container_width=True)
                img_bytes = uploaded_file.read() # Get bytes using .read() as per original code

            with col2:
                st.subheader("Analysis Results")
                temp_image_path = None # Initialize path variable
                try:
                    with st.spinner("🔎 Analyzing image..."):
                        # 1. Prepare Image
                        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        # Convert to OpenCV format for annotation
                        image_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                        # Optional Resizing (Keep commented out unless needed)
                        # max_size = (1024, 1024); pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
                        # image_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                        # Save image temporarily for Roboflow predict method
                        temp_image_path = f"temp_{uuid.uuid4()}.jpg"
                        pil_image.save(temp_image_path)
                        logger.info(f"Temp image saved: {temp_image_path}")

                        # 2. Run Prediction Directly using the loaded Roboflow model
                        logger.info(f"Running Roboflow prediction (v{ROBOFLOW_MODEL_VERSION})...")
                        result = roboflow_model.predict(temp_image_path, confidence=CONFIDENCE_THRESHOLD, overlap=OVERLAP_THRESHOLD).json()
                        logger.info("Prediction completed.")

                        # Check for errors within the Roboflow response
                        if "error" in result:
                            error_msg = result.get("error", "Unknown Roboflow error")
                            # Use st.error to display error in Streamlit UI
                            st.error(f"Roboflow prediction failed: {error_msg}")
                            logger.error(f"Roboflow API Error: {error_msg}")
                            predictions = []
                        else:
                             predictions = result.get("predictions", [])

                        if not predictions:
                            # Use st.warning in Streamlit UI
                            st.warning("No objects identified.")
                        else:
                            # 3. Process Predictions for Supervision
                            logger.info(f"Found {len(predictions)} predictions.")
                            labels, xyxys, confidences = [], [], []
                            detected_objects_for_response = [] # Store info for display matching original format
                            for item in predictions:
                                xc, yc, w, h = item["x"], item["y"], item["width"], item["height"]
                                conf, lbl = item["confidence"], item["class"]
                                xmin, ymin, xmax, ymax = xc-w/2, yc-h/2, xc+w/2, yc+h/2
                                xyxys.append([xmin, ymin, xmax, ymax])
                                confidences.append(conf)
                                formatted_label = f"{lbl} ({conf * 100:.1f}%)"
                                labels.append(formatted_label)
                                # Store dict for display matching original backend response structure
                                detected_objects_for_response.append({"label": lbl, "confidence": conf})


                            detections = sv.Detections(
                                xyxy=np.array(xyxys),
                                confidence=np.array(confidences),
                                class_id=np.array(range(len(labels))) # Dummy IDs
                            )

                            # 4. Annotate Image using Supervision (Corrected logic)
                            box_annotator = sv.BoxAnnotator(thickness=2)
                            label_annotator = sv.LabelAnnotator() # Use default settings

                            # Draw boxes first
                            annotated_image_with_boxes = box_annotator.annotate(
                                scene=image_cv2.copy(),
                                detections=detections
                            )
                            # Draw labels on the image with boxes
                            final_annotated_image = label_annotator.annotate(
                                scene=annotated_image_with_boxes,
                                detections=detections,
                                labels=labels # Pass the formatted labels
                            )
                            logger.info("Image annotation completed.")

                            # 5. Display Annotated Image in Streamlit
                            st.image(final_annotated_image, channels="BGR", caption="Analysis Visualization", use_container_width=True)

                            # 6. Display Detected Labels (matching original structure if needed)
                            st.write("**Detected:**")
                            if detected_objects_for_response:
                                for obj_info in detected_objects_for_response:
                                     # Display similar to how backend response was handled
                                     label = obj_info.get('label', 'Unknown Object')
                                     confidence = obj_info.get('confidence')
                                     display_text = f"- **{label}**"
                                     if confidence:
                                         display_text += f" (Confidence: {confidence*100:.1f}%)"
                                     st.success(display_text)
                            # This 'else' should technically not be reachable if predictions is not empty
                            # else:
                            #    st.warning("No specific breeds identified (this message shouldn't appear).")

                except Exception as e:
                    # Display error in Streamlit UI
                    st.error(f"An error occurred during image analysis: {e}")
                    logger.error(f"Error (Identify Breed): {e}\n{traceback.format_exc()}")
                finally:
                    # Clean up temporary file reliably
                    if temp_image_path and os.path.exists(temp_image_path):
                        try:
                            os.remove(temp_image_path)
                            logger.info(f"Temp file deleted: {temp_image_path}")
                        except Exception as del_err:
                            logger.error(f"Error deleting temp file {temp_image_path}: {del_err}")
        else:
             # Kept your original info message
            st.info("Upload a clear image file (JPG, PNG) containing cattle to begin identification.")

# <<< --- END INTEGRATED IDENTIFY BREED SECTION --- >>>

# 6. Chatbot
elif selected_page == "Chatbot":
    st.title("🧑‍🌾 Kamdhenu AI Assistant")
    st.markdown("Ask questions about indigenous breeds, farming practices, health, schemes, etc.")
    st.markdown("---")

    if not gemini_model: # Check if model was initialized successfully
        st.error("Chatbot is currently unavailable. Please ensure the Google API Key is correctly configured in the .env file and the model is accessible.", icon="🚫")
    else:
        try:
            # Initialize translator (consider caching this)
            translator = Translator()

            # Initialize chat history in session state
            if "messages" not in st.session_state:
                st.session_state.messages = []
            if "chat_language" not in st.session_state:
                st.session_state.chat_language = "en" # Default language

            # Language Selection
            language_options = {"English": "en", "हिन्दी (Hindi)": "hi", "తెలుగు (Telugu)": "te", "தமிழ் (Tamil)": "ta", "ગુજરાતી (Gujarati)": "gu", "ਪੰਜਾਬੀ (Punjabi)": "pa"} # Added Punjabi
            # Get index of current language for default selection
            lang_keys = list(language_options.keys())
            lang_values = list(language_options.values())
            current_lang_index = lang_values.index(st.session_state.chat_language) if st.session_state.chat_language in lang_values else 0

            selected_language_name = st.selectbox(
                "Choose interaction language:",
                lang_keys,
                index=current_lang_index,
                key="chat_lang_select" # Add a key
            )
            st.session_state.chat_language = language_options[selected_language_name]
            lang_code = st.session_state.chat_language

            # Display past chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Get user input
            if prompt := st.chat_input(f"Ask your question in {selected_language_name}..."):
                # Add user message to state and display it
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # --- Get and process AI Response ---
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    # Use Streamlit's spinner for better UX
                    with st.spinner(f"Thinking in {selected_language_name}..."):
                        try:
                            # Translate user input to English for the model if needed
                            prompt_en = prompt
                            if lang_code != 'en':
                                try:
                                    translation = translator.translate(prompt, src=lang_code, dest='en')
                                    prompt_en = translation.text
                                except Exception as trans_in_err:
                                     st.warning(f"Could not translate input to English: {trans_in_err}. Using original input.", icon="⚠️")
                                     prompt_en = prompt # Fallback to original

                            # Construct a focused prompt for the LLM
                            contextual_prompt = f"""
                            You are 'Kamdhenu Sahayak', an AI assistant for Indian farmers and cattle rearers. Focus specifically on:
                            1. Indigenous Indian cattle breeds (like Gir, Sahiwal, Ongole, Tharparkar, Kankrej, Rathi, Hallikar, etc.): Their care, characteristics, milk yield, draft power, climate suitability, and conservation status.
                            2. Sustainable & Eco-Friendly Farming Practices relevant to India, especially those involving cattle: Manure management (composting, biogas), rotational grazing, water conservation for livestock, agroforestry/silvopasture for fodder and shade, organic farming principles for fodder crops.
                            3. Common Cattle Diseases in India: Recognizing symptoms, basic first aid/preventive measures (e.g., vaccination schedules, deworming), but **always strongly emphasize consulting a qualified veterinarian** for actual diagnosis and treatment. Do not provide specific drug dosages. Mention diseases like FMD, HS, BQ, Mastitis, Scours, Bloat.
                            4. Indian Government Schemes for Agriculture & Animal Husbandry: Briefly explain the purpose, key benefits, and general eligibility criteria for major central schemes (like RGM, NLM, KCC, PM-KUSUM related to biogas) and notable state schemes if specified by the user (though your knowledge might be limited). Direct users to official portals for details.
                            5. General cattle lifecycle management: Key nutritional needs and care during different stages (calf, heifer, pregnant, lactating, dry cow, bull).
                            6. Basic factors affecting cattle price/valuation (breed, age, health, milk yield, pregnancy status, pedigree), but state that actual market prices vary greatly. Avoid giving specific price predictions.

                            Answer the following user question concisely and helpfully in a friendly, respectful tone appropriate for farmers.
                            Use simple language. If the question is completely unrelated to these topics, politely state that you specialize in Indian farming, particularly cattle care and sustainable practices, and cannot answer the unrelated query.
                            User question (potentially translated from {selected_language_name}): {prompt_en}
                            Respond *only* in {selected_language_name}. Ensure the response is well-formatted (e.g., use bullet points or short paragraphs for clarity).
                            """

                            # Generate the response using the initialized model
                            response = gemini_model.generate_content(contextual_prompt)

                            # --- ROBUST RESPONSE HANDLING ---
                            response_text_en = "" # Initialize empty response
                            try:
                                # Check candidates and parts for robustness (handles safety blocking)
                                if hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'content') and hasattr(response.candidates[0].content, 'parts') and response.candidates[0].content.parts:
                                    response_text_en = response.candidates[0].content.parts[0].text
                                else:
                                    # Handle cases where response structure is unexpected or blocked
                                    block_reason_msg = "Unknown reason."
                                    if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason'):
                                        block_reason_msg = f"Block Reason: {response.prompt_feedback.block_reason}."
                                    elif hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'finish_reason'):
                                         block_reason_msg = f"Finish Reason: {response.candidates[0].finish_reason}."

                                    st.warning(f"Warning: AI response may be empty or blocked. {block_reason_msg}", icon="⚠️")
                                    response_text_en = "I apologize, but I couldn't generate a complete response for that request. This might be due to safety filters or the query itself. Could you please try rephrasing?"

                            except ValueError as ve:
                                # This can happen if .text is accessed on a blocked response part
                                st.error(f"Error processing AI response (potentially blocked content): {ve}")
                                response_text_en = "I encountered an issue processing the response, possibly due to content filters. Please try again or rephrase your question."
                            except Exception as e_resp:
                                # Catch any other unexpected errors processing the response object
                                st.error(f"An unexpected error occurred while processing the AI response: {e_resp}")
                                response_text_en = "Sorry, an internal error occurred while processing the response."
                            # --- END ROBUST RESPONSE HANDLING ---


                            # Translate response back to the user's language if needed
                            final_response_text = response_text_en
                            if lang_code != 'en' and response_text_en: # Avoid translating empty strings or error messages
                                 try:
                                     # Ensure text isn't just the error message before translating
                                     if "I apologize" not in response_text_en and "I encountered an issue" not in response_text_en:
                                         final_response_translation = translator.translate(response_text_en, src='en', dest=lang_code)
                                         final_response_text = final_response_translation.text
                                     else:
                                         # Attempt to translate the apology/error message itself
                                          final_response_translation = translator.translate(response_text_en, src='en', dest=lang_code)
                                          final_response_text = final_response_translation.text
                                 except Exception as trans_err:
                                     st.error(f"Error translating response to {selected_language_name}: {trans_err}")
                                     # Fallback to English response with note
                                     final_response_text = f"(Translation Error) {response_text_en}"


                            # Display the final response
                            message_placeholder.markdown(final_response_text)
                            # Add response to session state
                            st.session_state.messages.append({"role": "assistant", "content": final_response_text})

                        except Exception as e:
                            # General error during API call or translation setup
                            st.error(f"Error generating response: {e}")
                            error_msg = f"Sorry, I encountered an error processing your request in {selected_language_name}. Please try again or ask differently."
                            # Translate error message if possible
                            try:
                                if lang_code != 'en':
                                    error_msg = translator.translate(error_msg, src='en', dest=lang_code).text
                            except Exception:
                                pass # Keep English error if translation fails
                            message_placeholder.markdown(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})

        except Exception as e:
            # Error initializing translator etc.
            st.error(f"Chatbot failed to initialize: {e}. Please check configuration.")


# 7. Price Trends & Calculator
elif selected_page == "Price Trends":
    st.title("📈 Cattle Price Trends & Valuation Estimator")
    st.markdown("Analyze illustrative historical price data and estimate the potential value of your cattle.")
    st.markdown("---")

    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        # Section for Price Trends Chart
        st.subheader("📈 Historical Average Price Trends (Illustrative Data)")
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='price_trends'")
            if cursor.fetchone():
                cursor.execute("SELECT year, average_price FROM price_trends ORDER BY year ASC")
                data = cursor.fetchall()
                if data:
                    df_trends = pd.DataFrame(data, columns=["Year", "Average Price (INR)"])
                    df_trends = df_trends.set_index("Year")
                    st.line_chart(df_trends)
                    # Metrics below chart
                    if len(df_trends) > 1:
                        latest_price = df_trends["Average Price (INR)"].iloc[-1]
                        previous_price = df_trends["Average Price (INR)"].iloc[-2]
                        price_change = latest_price - previous_price
                        st.metric(label="Latest Avg Price", value=f"₹{latest_price:,.0f}", delta=f"₹{price_change:,.0f} vs Previous Year")
                    elif len(df_trends) == 1:
                         st.metric(label="Latest Avg Price", value=f"₹{df_trends['Average Price (INR)'].iloc[-1]:,.0f}")
                else:
                    st.info("No historical price data found in the database to display trends.")
            else:
                 st.warning("Database table 'price_trends' not found.")
        except sqlite3.Error as e:
            st.error(f"Database error fetching price trends: {e}")

        st.markdown("---")
        # Section for Price Calculator
        st.subheader("📊 Cattle Valuation Estimator")
        st.caption("Provides a rough estimate. Actual market value depends on many local factors.")

        col1, col2, col3 = st.columns(3)
        with col1:
            breed_list = sorted([b["name"] for b in CATTLE_BREEDS_DATA]) + ["Murrah (Buffalo)", "Crossbred", "Other"]
            breed = st.selectbox("Select Breed", breed_list, key="calc_breed_val")
        with col2:
            age = st.number_input("Age (Years)", min_value=0.5, max_value=25.0, value=4.0, step=0.5, key="calc_age_val")
        with col3:
             weight = st.number_input("Approx. Weight (Kg)", min_value=50, max_value=1200, value=350, step=10, key="calc_weight_val")

        col4, col5, col6 = st.columns(3)
        with col4:
            milk_yield = st.number_input("Avg. Daily Milk Yield (Liters)", min_value=0.0, max_value=50.0, value=8.0, step=0.5, key="calc_milk_val", help="Enter 0 if not applicable/male")
        with col5:
             health_status = st.selectbox("Overall Health Condition", ["Excellent", "Good", "Fair", "Poor"], key="calc_health_val")
        with col6:
            is_pregnant = st.checkbox("Currently Pregnant?", key="calc_pregnant_val")

        if st.button("Estimate Valuation", type="primary", key="btn_estimate"):
            # (Using the improved calculation logic from previous step)
            base_price = 30000
            breed_factors = {
                "Gir": 1.8, "Sahiwal": 1.9, "Red Sindhi": 1.7, "Tharparkar": 1.6,
                "Ongole": 1.5, "Kankrej": 1.6, "Rathi": 1.5, "Murrah (Buffalo)": 2.0,
                "Crossbred": 1.2, "Punganur": 1.0, "Amrit Mahal": 1.3, "Hallikar": 1.4,
                "Deoni": 1.4, "Krishna Valley": 1.4, "Malnad Gidda": 1.1
            }
            base_price *= breed_factors.get(breed, 1.0)
            if 2.5 <= age <= 8: age_factor = 1.15
            elif age < 2.5: age_factor = 0.8 + (age / 5)
            else: age_factor = max(0.6, 1.1 - (age - 8) * 0.05)
            base_price *= age_factor
            weight_factor = 1.0 + (min(weight, 600) - 300) * 0.001
            base_price *= weight_factor
            if milk_yield > 1: # Give boost only if > 1 liter
                 milk_factor = 1.0 + (milk_yield * 0.05)
                 base_price *= milk_factor
            health_factors = {"Excellent": 1.1, "Good": 1.0, "Fair": 0.85, "Poor": 0.6}
            base_price *= health_factors.get(health_status, 0.9)
            if is_pregnant: base_price *= 1.1
            estimated_price = max(15000, base_price) # Increased floor price slightly

            st.success(f"Estimated Valuation Range: ₹ {estimated_price * 0.85:,.0f} - ₹ {estimated_price * 1.15:,.0f}")
            st.caption("Disclaimer: This is an indicative range. Actual market prices vary significantly based on location, pedigree, specific traits, current demand, and negotiation.")

        #conn.close()
    else:
        st.error("Database connection failed. Cannot load Price Trends & Calculator.")


# 8. Disease Diagnosis
elif selected_page == "Diagnosis":
    st.title("🩺 Disease Diagnosis Assistant (Beta)")
    st.warning("**Disclaimer:** This tool provides potential suggestions based on symptom matching with database entries. It is **NOT** a substitute for professional veterinary diagnosis. **Always consult a qualified veterinarian** for accurate diagnosis and treatment of sick animals.", icon="⚠️")
    st.markdown("---")

    symptoms_input = st.text_area("Enter Observed Symptoms (comma-separated, e.g., high fever, difficulty breathing, reduced milk yield, limping):", height=100)

    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        if st.button("Suggest Potential Diseases", type="primary", key="btn_diagnose"):
            if symptoms_input and symptoms_input.strip():
                symptoms_list = [s.strip().lower() for s in symptoms_input.split(',') if s.strip() and len(s.strip()) > 2] # Ignore very short inputs
                if not symptoms_list:
                     st.warning("Please enter valid symptoms (minimum 3 characters each) separated by commas.")
                else:
                    st.write(f"Searching for matches for symptoms: **{', '.join(symptoms_list)}**")
                    # Build query safely
                    query_parts = []
                    params = []
                    for symptom in symptoms_list:
                        query_parts.append("LOWER(symptoms) LIKE ?")
                        params.append(f"%{symptom}%")

                    # Query using OR logic, could be refined with AND or scoring for better results
                    query = f"""
                    SELECT detected_disease, suggested_treatment, severity, symptoms
                    FROM disease_diagnosis
                    WHERE {' OR '.join(query_parts)}
                    ORDER BY RANDOM() LIMIT 5
                    """ # Simple OR matching, limit results

                    try:
                        cursor.execute("SELECT 1 FROM disease_diagnosis LIMIT 1") # Check table exists
                        cursor.execute(query, params)
                        results = cursor.fetchall()

                        if results:
                            st.subheader("Potential Matches Based on Symptoms:")
                            st.caption("Note: A disease may appear if *any* entered symptom matches its typical list.")
                            for disease, treatment, severity, db_symptoms in results:
                                # Highlight matching symptoms
                                matched_symptoms_display = db_symptoms
                                for user_symptom in symptoms_list:
                                    matched_symptoms_display = matched_symptoms_display.replace(user_symptom, f"**{user_symptom}**")

                                sev_color = "blue"
                                sev_icon = "ℹ️"
                                if severity and severity.lower() == 'high':
                                    sev_color = "red"
                                    sev_icon = "🚨"
                                elif severity and severity.lower() == 'medium':
                                    sev_color = "orange"
                                    sev_icon = "⚠️"

                                st.markdown(f"<font color='{sev_color}'> **{sev_icon} {disease}** (Severity: {severity or 'Unknown'})</font>", unsafe_allow_html=True)
                                with st.container(border=True):
                                    st.markdown(f"**Typical Symptoms Include:** {matched_symptoms_display}", unsafe_allow_html=True)
                                    st.markdown(f"**General Suggested Action:** {treatment}")
                                    st.markdown(f"*Consult a vet immediately for confirmation and specific treatment plan.*")

                        else:
                            st.warning("No common diseases strongly matched the entered symptoms in our database. Please consult a veterinarian for an accurate diagnosis.")

                    except sqlite3.OperationalError:
                         st.error("Database table 'disease_diagnosis' not found or inaccessible.")
                    except sqlite3.Error as e:
                        st.error(f"Database error during diagnosis lookup: {e}")
                    except Exception as e:
                         st.error(f"An unexpected error occurred: {e}")
            else:
                st.warning("Please enter symptoms before searching.")
         #conn.close()
    else:
         st.error("Database connection failed. Cannot load Disease Diagnosis Assistant.")


# 9. Government Schemes
elif selected_page == "Govt Schemes":
    st.title("🏛️ Government Schemes Information Hub")
    st.markdown("Explore Central and State government schemes relevant to agriculture and animal husbandry.")
    st.markdown("---")

    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        available_regions = ["All India / Central"] # Default option
        available_types = ["All Types"]

        try:
             # Fetch distinct regions and types for filters
            cursor.execute("SELECT DISTINCT region FROM government_schemes WHERE region IS NOT NULL AND region != '' ORDER BY region ASC")
            available_regions.extend([region[0] for region in cursor.fetchall()])
            cursor.execute("SELECT DISTINCT type FROM government_schemes WHERE type IS NOT NULL AND type != '' ORDER BY type ASC")
            available_types.extend([type[0] for type in cursor.fetchall()])

        except sqlite3.Error as e:
            st.error(f"Error fetching filter options from database: {e}")

        # Filtering options side-by-side
        col1, col2 = st.columns(2)
        with col1:
            selected_region = st.selectbox("Filter by Region:", available_regions, index = 0) # Default to All India
        with col2:
            selected_type = st.selectbox("Filter by Scheme Type:", available_types, index=0) # Default to All Types

        # Search Box
        search_term = st.text_input("🔍 Search by Scheme Name or Keyword:", placeholder="e.g., Kisan Credit Card, NLM, Subsidy...")

        try:
            # Build query dynamically and safely
            query = "SELECT name, details, url, region, type FROM government_schemes WHERE 1=1"
            params = []

            if selected_region != "All India / Central":
                query += " AND region = ?"
                params.append(selected_region)
            elif selected_region == "All India / Central":
                 query += " AND (region = ? OR region IS NULL OR region = '' OR region LIKE '%Central%')" # Broader match for Central
                 params.append(selected_region)

            if selected_type != "All Types":
                query += " AND type = ?"
                params.append(selected_type)

            if search_term:
                query += " AND (name LIKE ? OR details LIKE ?)"
                params.append(f"%{search_term}%")
                params.append(f"%{search_term}%")

            query += " ORDER BY name ASC"

            cursor.execute("SELECT 1 FROM government_schemes LIMIT 1") # Check table exists
            cursor.execute(query, params)
            schemes = cursor.fetchall()

            # Display results
            st.markdown("---")
            st.subheader(f"Found {len(schemes)} Matching Schemes:")
            if schemes:
                for name, details, url, region_db, type_db in schemes:
                     meta_info = []
                     if region_db: meta_info.append(f"📍 {region_db}")
                     if type_db: meta_info.append(f"🏷️ {type_db}")
                     # Clean up details preview
                     details_preview = (details[:200] + '...') if len(details) > 200 else details

                     with st.expander(f"**{name}** {' | '.join(meta_info) if meta_info else ''}"):
                        st.caption(f"Type: {type_db or 'N/A'} | Region: {region_db or 'Central/Multiple'}")
                        st.markdown(details) # Show full details inside expander
                        if url and url.strip().startswith("http"):
                            st.link_button("🔗 Official Source / Learn More", url, help=f"Visit official page for {name}")
                        elif url and url.strip():
                            st.caption(f"Reference/Source: {url.strip()}")
            elif search_term or selected_region != "All India / Central" or selected_type != "All Types":
                st.info(f"No schemes found matching your specific criteria. Try broadening your search filters.")
            else:
                 st.info("No schemes found in the database.")


        except sqlite3.OperationalError:
             st.error("Database table 'government_schemes' not found.")
        except sqlite3.Error as e:
             st.error(f"Error fetching schemes: {e}")
        except Exception as e:
             st.error(f"An unexpected error occurred: {e}")

        #conn.close()
    else:
        st.error("Database connection failed. Cannot load Government Schemes.")


# 10. Lifecycle Management
elif selected_page == "Lifecycle Management":
    st.title("🔄 Cattle Lifecycle Management Guide")
    st.markdown("Essential care and management practices for cattle at different life stages.")
    st.markdown("---")

    # (Using the expanded lifecycle_stages dictionary from the previous step)
    lifecycle_stages = {
        "Calf (0-6 months)": {
            "image": "images/calf.jpeg",
            "focus": "Immunity, Growth Start, Weaning",
            "details": [
                "**Colostrum:** Critical! Feed 10% of body weight within 2-4 hours of birth.",
                "**Housing:** Clean, dry, warm, draft-free pen. Individual housing initially recommended.",
                "**Feeding:** High-quality milk replacer or whole milk. Introduce calf starter feed (18-20% Protein) from day 3-4.",
                "**Water:** Fresh, clean water available from day 1.",
                "**Health:** Navel disinfection, monitor for scours & pneumonia. Deworming & initial vaccinations (consult vet).",
                "**Weaning:** Gradual process around 8-10 weeks, based on starter intake (e.g., eating >1 kg/day).",
            ]
        },
        "Heifer (6-24 months)": {
             "image": "images/heif.jpeg",
             "focus": "Growth, Sexual Maturity, Breeding Preparation",
             "details": [
                "**Nutrition:** Balanced ration for steady growth (avoid fattening). Target ~60-65% of mature body weight at first breeding.",
                "**Forage:** Good quality green fodder & hay form the base.",
                "**Concentrate:** Supplement as needed based on forage quality and growth rate (14-16% Protein).",
                "**Minerals:** Provide balanced mineral mixture.",
                "**Health:** Regular deworming & booster vaccinations. Monitor for parasites.",
                "**Breeding:** Observe for heat cycles starting around 9-15 months. Breed based on weight & age (typically 15-18 months). Use AI or tested bull.",
             ]
        },
        "Pregnant Cow/Heifer": {
             "image": "images/preg.jpeg",
             "focus": "Fetal Growth, Udder Development, Calving Preparation",
             "details": [
                "**Early/Mid Gestation (Months 1-6):** Maintain good body condition. Nutrition similar to dry cow or late heifer.",
                "**Late Gestation (Months 7-9):** Nutrient needs increase significantly (esp. protein, energy, calcium, phosphorus) for fetal growth & colostrum production. Provide ~25% extra energy.",
                "**Feeding:** High-quality forage + appropriate concentrate supplement. Avoid sudden feed changes.",
                "**Minerals:** Crucial! Ensure adequate Calcium, Phosphorus, Selenium, Vit E.",
                "**Health:** Monitor body condition. Booster vaccinations (e.g., against scours pathogens) 4-6 weeks before calving.",
                "**Management:** Avoid stress. Move to clean, comfortable calving pen 1-2 weeks before expected date.",
             ]
        },
        "Lactating Cow": {
             "image": "images/lac.jpeg",
             "focus": "Milk Production, Health Maintenance, Re-breeding",
             "details": [
                 "**Nutrition:** Highest demand! Feed based on milk yield, stage of lactation, and body condition.",
                 "**Energy & Protein:** Key drivers of milk production. High-quality forages + balanced concentrates (16-18% Protein).",
                 "**Water:** Crucial! Need 4-5 liters water per liter of milk produced + maintenance needs.",
                 "**Minerals:** Especially Calcium & Phosphorus. Provide free-choice mineral mix.",
                 "**Milking:** Hygienic practices (clean udder, hands, equipment). Consistent milking times.",
                 "**Health:** Monitor for mastitis (check milk), lameness, metabolic diseases (ketosis, milk fever - esp. early lactation).",
                 "**Breeding:** Aim to re-breed within 60-90 days post-calving for optimal calving interval.",
             ]
        },
         "Dry Cow (Non-lactating period)": {
             "image": "images/dry.jpeg",
             "focus": "Udder Rest & Regeneration, Fetal Growth (late dry), Preparing for Lactation",
             "details": [
                 "**Duration:** Typically 45-60 days before expected calving date.",
                 "**Nutrition:** Lower requirements than lactation. Maintain body condition (Score 3.0-3.5). Avoid getting fat.",
                 "**Feeding:** Primarily good quality forage. Low or no concentrate initially, increase slightly in the last 2-3 weeks ('transition period').",
                 "**Minerals:** Adjust mineral mix, especially Calcium (reduce slightly in early dry period, increase pre-calving).",
                 "**Health:** Ideal time for treating subclinical mastitis (Dry Cow Therapy - consult vet). Monitor overall health.",
                 "**Management:** Separate from milking herd if possible. Provide comfortable housing.",
             ]
        },
         "Bull / Breeding Male": {
             "image": "images/bull.jpeg",
             "focus": "Maintaining Libido & Fertility, Soundness, Safe Handling",
             "details": [
                 "**Nutrition:** Balanced diet to maintain good condition (not fat). Requirements vary based on age and breeding activity.",
                 "**Feeding:** Good forage + moderate concentrate (12-14% Protein). Ensure adequate minerals (Zinc, Selenium).",
                 "**Exercise:** Provide adequate space for movement.",
                 "**Health:** Regular checks for lameness, reproductive organ health. Annual Breeding Soundness Exam recommended.",
                 "**Management:** Handle with extreme caution (use proper facilities). Monitor breeding activity and libido.",
                 "**Biosecurity:** Test for reproductive diseases. Quarantine new arrivals.",
             ]
         }
    }

    selected_stage = st.selectbox("Select Lifecycle Stage to View Management Tips:", list(lifecycle_stages.keys()))

    if selected_stage:
        stage_info = lifecycle_stages[selected_stage]
        st.subheader(f"Focus for {selected_stage}: {stage_info['focus']}")

        col1, col2 = st.columns([1, 2]) # Adjust column ratio if needed
        with col1:
             display_image(stage_info.get("image"), caption=f"{selected_stage}") # Display image

        with col2:
            st.markdown("**Key Considerations & Actions:**")
            for point in stage_info["details"]:
                st.markdown(f"- {point}")

    st.markdown("---")
    st.info("ℹ️ These guidelines provide a general overview. Specific needs vary significantly based on breed, climate, available feed resources, and individual animal health. Professional consultation is recommended for optimal management.")


# --- Footer ---
st.markdown("---")
st.caption("Kamdhenu Program App v1.1 © 2024 | Empowering Sustainable Indian Farming")
