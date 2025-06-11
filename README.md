# Reviving the Indian Cow Breed: A Sustainable Future 🐄🇮🇳

Welcome to the **Kamdhenu Program** digital platform! This project is dedicated to empowering Indian farmers and cattle rearers by providing a comprehensive suite of tools and information focused on the conservation of indigenous cattle breeds and the adoption of sustainable agricultural practices. Leveraging modern technologies like AI, geolocation, and multilingual chat, we aim to enhance farm productivity, improve animal health, and contribute to a resilient agricultural future for India.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Live_App-Kamdhenu-brightgreen?logo=streamlit)](https://kamdhenu.streamlit.app/)
---
## 🚀 Live Demo

Experience the Kamdhenu platform in action. No setup required!

👉 **Try the App Now**: [https://kamdhenu.streamlit.app/](https://kamdhenu.streamlit.app/)

## ✨ Key Features

This application integrates multiple modules accessible through an intuitive web interface:

- **🏠 Home Dashboard**: Quick access to all tools and insights.
- **🧬 Indigenous Breed Info**: Rich profiles of Indian cattle breeds with search and filtering.
- **💖 Breeding Program**: Suggest pairs, record offspring, and track genetics.
- **🌱 Eco-Friendly Practices**: Sustainable agriculture tips like composting, water reuse, IPM, etc.
- **🎨 AI Breed Detection**: Upload a cattle image to get breed prediction using a Roboflow model.
- **🧠 Skin Disease Detector**: TensorFlow model classifies skin conditions (bacterial, fungal, or healthy).
- **🗨️ Multilingual AI Chatbot**: Google Gemini-powered assistant supporting 6+ Indian languages.
- **📈 Price Trends & Valuation**: See regional pricing data and estimate cattle value.
- **🛒 Marketplace**: List and browse cattle or machinery for sale or rent with image uploads.
- **📋 Health Records & Alerts**: Vaccination logs, symptom tracking, and smart reminders.
- **🧾 Milk Yield Log**: Track milk yield, fat/SNF percentage, and generate reports.
- **📍 Geolocation**: Converts addresses to coordinates for regional suggestions.
- **🤝 Community Forum**: Share questions, get expert replies, and build a knowledge network.
- **🎁 Adoption & Donation**: Sponsor cattle, support Gaushalas, and log donor history.
- **📤 Report Export**: Generate PDF summaries using ReportLab.
- **🔥 Firestore Integration**: Option to sync data to Firebase Firestore for cloud scalability.

---

## 🧱 System Architecture

The following diagram provides a high-level overview of how the Kamdhenu App is structured and how its components interact.

![Kamdhenu System Architecture](https://github.com/Yasaswini-ch/_Reviving_the_Indian_Cow_Breed_A_Sustainable_Future_/blob/main/assets/Screenshot%202025-06-11%20211226.png?raw=true)

> 📌 **Explanation:**
>
> - **Users** (Farmer, Buyer, Public) interact with the application through a Streamlit-based web interface.
> - The **Core Logic** handles user authentication, page routing, UI rendering, and data preprocessing.
> - The app makes API calls to external services:
>   - 🌦️ **Open-Meteo** for weather data.
>   - 🌐 **Google Translate** (via the `deep_translator` library) for multilingual support.
>   - 🤖 **Google Gemini** for the generative AI chatbot.
> - **Data is stored in:**
>   - `Cows.db` (SQLite) for core user, farm, and listing data in the local setup.
>   - A JSON file for community forum discussions.
>   - Local `uploaded_images/` directory for cattle, disease, and machinery images.
> - **Machine Learning Models:**
>   - 🧠 **Roboflow** for cloud-based AI breed identification.
>   - 🧪 **TensorFlow** for offline skin disease prediction.

---

## 🛠️ Technology Stack

| Layer             | Technology Used                                                      |
|------------------|-----------------------------------------------------------------------|
| Frontend UI       | Streamlit                                                            |
| ML Models         | TensorFlow (Skin Disease), Keras, Roboflow API (Breed ID)            |
| Backend           | SQLite (Local DB), Firebase Firestore (Cloud DB Option)              |
| Chatbot           | Google Generative AI (Gemini via `google-generativeai`)              |
| Translation       | Google Translate API, Deep Translator                                |
| Geolocation       | Geopy (OpenStreetMap), Google Maps API (planned/optional)            |
| Payment System    | UPI ID Handling, Google Pay Integration (for donations/adoptions)    |
| External Services | Open-Meteo Weather API, Google APIs                                  |
| Image Processing  | PIL (Image handling), OpenCV, Supervision                            |
| Export            | ReportLab (PDF generation), Pandas (Excel exports)                   |
| Auth & Security   | bcrypt password hashing                                              |
| Storage           | Uploaded Files (images), Firebase (Firestore, optional), Local FS    |
| Deployment        | Streamlit Cloud (Live: [kamdhenu.streamlit.app](https://kamdhenu.streamlit.app/)) |


---

## 🏋️‍♂️ Repository Structure
```
kamdhenu-app-main/
├── model/                          # Contains the machine learning model files
├── uploaded_images/                # Directory to store user-uploaded images
├── .env                            # Environment variables (should be in .gitignore)
├── LICENSE                         # Project license file
├── README.md                       # Project documentation and instructions
├── app.py                          # The main application file (e.g., Streamlit, Flask)
├── apt.txt                         # System-level dependencies for deployment (e.g., on Streamlit Cloud)
├── community_forum.py              # Backend logic for a community forum feature
├── Cows.db                         # SQLite database file for storing data
├── disease_detector.py             # Module for the core disease detection logic
├── disease_diagnosis.py            # Module for providing diagnosis based on detection
├── indian-cow-breed-....json       # Firebase Admin SDK key (should be in .gitignore)
├── migrate_sqlite_to_firestore.py  # Script to migrate data from local SQLite to Firebase Firestore
├── requirements.txt                # Python package dependencies (for pip install -r)
├── runtime.txt                     # Specifies the Python runtime version for deployment
├── setup_database.py               # Script to initialize or set up the SQLite database
├── temp_image.jpg                  # A temporary or placeholder image used by the app
└── translation_utils.py            # Utility functions for handling language translation
```

---

## 💂️ Getting Started

### **1. Prerequisites**

- Python 3.10 or higher installed.
- Git installed.
- Access to a terminal or command prompt.

### **2. Clone the Repository**

```bash
git clone https://github.com/Yasaswini-ch/_Reviving_the_Indian_Cow_Breed_A_Sustainable_Future_
cd kamdhenu-app-main
```

### **3. Create and Activate Virtual Environment**

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### **4. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **5. Configure Environment Variables**

Create a `.env` file in the project root directory and add:

```env
GOOGLE_API_KEY=YOUR_GOOGLE_GEMINI_API_KEY
ROBOFLOW_API_KEY=YOUR_ROBOFLOW_API_KEY
```

### **6. Set Up the Database**

```bash
python setup_database.py
```

### **7. Running the Application**
#### Start the Streamlit Frontend**
### **8. (Optional) Connect to Firebase Firestore**

Use Firestore to store data in the cloud instead of locally.

---

### **8.1 Create Firebase Project**

- Go to [Firebase Console](https://console.firebase.google.com/)
- Click **“Add project”** and follow the setup steps.

---

### **8.2 Enable Firestore Database**

- Navigate to **Build > Firestore Database**
- Click **“Create database”**, choose **Test** or **Production mode**

---

### **8.3 Generate Service Account Key**

- Go to **Project Settings > Service Accounts**
- Click **“Generate new private key”**
- Download the `.json` key file

---

### **8.4 Place the Key File**

- Rename it to:
  ```txt
  indian-cow-breed-firebase-adminsdk.json

- Move it to your project’s **root directory**.

### 8.5 Run Migration Script  
- Execute the following Python script to migrate data from SQLite to Firestore:
```bash
python migrate_sqlite_to_firestore.py
```
✅ Done! Your SQLite data is now synced to Firestore. 

```bash
streamlit run app.py
```

---
## ☁️ Deployment

To deploy on **Streamlit Community Cloud**:

- Ensure `requirements.txt` is updated.
- Use `secrets.toml` instead of `.env`:

```toml
# .streamlit/secrets.toml
GOOGLE_API_KEY = "YOUR_GOOGLE_GEMINI_API_KEY_HERE"
ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY_HERE"
```

- Modify `app.py`:

```python
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
ROBOFLOW_API_KEY = st.secrets.get("ROBOFLOW_API_KEY")
```

- Push to GitHub/GitLab and connect to Streamlit Cloud.
  
---

## 🧠 How It Works

The Kamdhenu App is designed with a singular vision: **to preserve and promote India’s indigenous cattle heritage** while empowering rural communities with technology. It serves as a digital extension of the **Kamdhenu ideology**, aligning ancient wisdom with modern innovation.

---

### 🐄 1. Promote Indigenous Breeds

- The app offers detailed profiles of native Indian cattle like Gir, Sahiwal, Ongole, etc.
- Farmers can:
  - Learn about breed characteristics (milk yield, lifespan, region)
  - Identify their cattle using AI-based image recognition
  - Make informed breeding decisions to maintain genetic purity

🎯 **Impact:** Encourages preservation of native breeds over commercial hybrids.

---

### 🌱 2. Encourage Sustainable Practices

- Farmers get easy access to:
  - Organic manure techniques
  - Water conservation methods
  - Biogas and composting guidance
  - Natural disease prevention tips

🎯 **Impact:** Reduces chemical dependency and aligns with Vedic farming principles.

---

### 💬 3. Empower Through Knowledge & Language

- A multilingual chatbot powered by AI helps farmers:
  - Ask questions about breed care, health, government schemes
  - Get responses in **their own language** (Hindi, Telugu, Tamil, Punjabi, etc.)

🎯 **Impact:** Makes advanced knowledge accessible to all, regardless of literacy or language barriers.

---

### 🛒 4. Support Gaushalas & Local Farmers

- Users can:
  - View adoptable cows and donate to shelters
  - Support campaigns like fodder supply, medical care, etc.
  - Buy/sell cattle and farming equipment ethically

🎯 **Impact:** Connects supporters and farmers to conserve cows and promote self-reliance.

---

### 📍 5. Localized & Farmer-Centric

- The app adapts to the **farmer’s region**:
  - Translates data to local language
  - Offers weather-based suggestions
  - Logs cattle health and productivity over time

🎯 **Impact:** Ensures every farmer gets guidance tailored to their land and livestock.

---

> 🙏 **Kamdhenu is not just a digital tool — it's a movement.**  
This app ensures that the principles of **cow protection, ethical farming, and farmer dignity** thrive in the digital era.


## 💁️ Future Scope

- 📱 Native mobile app with offline access and notifications  
- 🛰️ IoT integration for live cattle health tracking  
- 🛒 Advanced marketplace with transport, insurance, and verification  
- 🎓 Farmer learning modules with regional certification  
- 💸 Integrated UPI/Google Pay for donations and purchases  
- 🧾 Government scheme auto-detection and subsidy claim guidance  
- 🌐 NGO, Gaushala, and state-level dashboard onboarding  
- 📊 Real-time data analytics for breed trends and health alerts

---



## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

Special thanks to the creators of the libraries used (Streamlit, FastAPI, Roboflow, etc.).
<p align="center">
🧑‍🌾🐄 Let’s blend tradition with technology to revive India’s native cattle breeds and empower our farmers.
</p>

