# Reviving the Indian Cow Breed: A Sustainable Future 🐄🇮🇳

Welcome to the **Kamdhenu Program** digital platform! This project is dedicated to empowering Indian farmers and cattle rearers by providing a comprehensive suite of tools and information focused on the conservation of indigenous cattle breeds and the adoption of sustainable agricultural practices. Leveraging modern technologies like AI, we aim to enhance farm productivity, improve animal health, and contribute to a resilient agricultural future for India.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Key Features

This application integrates multiple modules accessible through an intuitive web interface:

1.  **🏠 Home:** Overview of the Kamdhenu Program and key features.
2.  **🧬 Breed Info:** Detailed profiles of various indigenous Indian cattle breeds, including images, characteristics (milk yield, strength, lifespan), region, search, and filtering capabilities. [Learn About Indigenous Breeds](https://bodhishop.in/blogs/news/indigenous-cow-breeds-of-india-popular-and-less-popular-ones?srsltid=AfmBOor5RNv-St9xH031N7eyvSX7o696XeluKNpoyrOnTknGk4eirJgL)
3.  **💖 Breeding Program Manager:** Tools to suggest potential breeding pairs based on goals (e.g., milk yield, purity), log suggestions, and view offspring records (requires database population).
4.  **🌱 Eco-Friendly Practices Guide:** Information on sustainable techniques suitable for cattle rearers and farmers, such as manure management, composting, biogas production, water conservation, rotational grazing, agroforestry, and IPM. Includes basic sustainability calculators.
5.  **🎨 AI Breed Identification:** Upload a cattle image to get an AI-powered breed prediction using a custom-trained Roboflow model (via FastAPI backend). Displays annotated images with bounding boxes.
6.  **💬 AI Chatbot Assistant:** An interactive, multilingual chatbot powered by Google Gemini API to answer farmer queries on breeds, practices, health, schemes, etc. Supports English, Hindi, Telugu, Tamil, Gujarati, and Punjabi.
7.  **📈 Price Trends & Valuation:** Visualize historical cattle price data and use a calculator to estimate the potential valuation of cattle based on various factors (breed, age, health, milk yield).
8.  **💪 Disease Diagnosis Assistant (Beta):** Enter observed symptoms to get potential disease suggestions based on a knowledge base. **Disclaimer:** This is NOT a substitute for professional veterinary advice.
9.  **🏦 Government Schemes Hub:** A searchable database of relevant Central and State government schemes for agriculture and animal husbandry, including details and links.
10. **🛠️ Lifecycle Management Guide:** Provides essential care and management tips for cattle across different life stages (Calf, Heifer, Pregnant, Lactating, Dry Cow, Bull).

---

## 🛠️ Technology Stack

### **Frontend:**
- [Streamlit](https://streamlit.io/)
- [streamlit-option-menu](https://github.com/victoryhb/streamlit-option-menu)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)

### **Database:**
- [SQLite](https://www.sqlite.org/index.html)

### **AI / ML APIs:**
- [Google Gemini API](https://ai.google.dev/) (via `google-generativeai`)
- [Roboflow API](https://roboflow.com/) (via `roboflow` library)

### **Data Handling & Processing:**
- [Pandas](https://pandas.pydata.org/)
- [Pillow (PIL)](https://python-pillow.org/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [Roboflow Supervision](https://supervision.roboflow.com/)

### **Translation & Utilities:**
- [googletrans](https://pypi.org/project/googletrans/) (Unofficial API)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [Requests](https://requests.readthedocs.io/en/latest/)

### **Language:**
- Python 3.10+

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


## 💂️ How It Works (Detailed Working)

1. **User Interaction:** The user accesses the Streamlit interface to explore breeds, ask queries, or perform tasks such as AI-based breed identification.
2. **Backend Processing:** User requests are sent to the FastAPI backend, where AI models (Roboflow for breed identification, Google Gemini for chatbot) process the data.
3. **Database Querying:** If the request involves breed details, government schemes, or price trends, the SQLite database retrieves relevant information.
4. **AI Model Execution:** For image-based breed identification, the image is processed using OpenCV, and the model predicts the breed.
5. **Result Display:** The processed information is returned to the Streamlit UI, displaying results with visual aids like charts, images, or formatted text.

---

## 🚀 Live Deployment

Access the deployed Streamlit app here: [Kamdhenu App](<https://kamadhenu.streamlit.app/>)

---

## 💁️ Future Scope

- **Mobile Application:** Develop an Android/iOS app for better accessibility.
- **Advanced AI Diagnostics:** Enhance disease prediction using deep learning.
- **Automated Cattle Pricing:** AI-based valuation using live market data.
- **Integration with Government Databases:** Real-time updates on schemes and subsidies.
- **IoT-based Health Monitoring:** Sensors for real-time cattle health tracking.

---

## 🏋️‍♂️ Repository Structure
```
kamdhenu-app-main/
│── app.py                   # Main Streamlit frontend
│── .streamlit/              # streamlit
  │── secrets.toml
│── setup_database.py        # Database setup script
│── requirements.txt         # Dependencies
│── .env                     # Environment variables (not committed)
│── README.md                # Project documentation
|── Cows.db                   # Database
```

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

Special thanks to the creators of the libraries used (Streamlit, FastAPI, Roboflow, etc.).

