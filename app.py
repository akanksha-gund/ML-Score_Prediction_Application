import pickle
import numpy as np
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor", page_icon="🎓", layout="centered"
)

# Custom CSS for gorgeous aesthetic styling, glassmorphism, and gradients
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    }
    .banner-container {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 25px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .banner-img {
        width: 100%;
        height: 200px;
        object-fit: cover;
        opacity: 0.65;
        filter: brightness(0.9);
    }
    .banner-text {
        position: absolute;
        bottom: 20px;
        left: 20px;
        right: 20px;
    }
    .banner-title {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    .banner-subtitle {
        font-size: 14px;
        color: #cbd5e1;
        margin-top: 4px;
    }
    /* Style form buttons and containers */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #06b6d4 0%, #6366f1 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem;
        border-radius: 0.5rem;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Load the trained KNN model
MODEL_PATH = "knn.pkl"


@st.cache_resource
def load_model():
  try:
    with open(MODEL_PATH, "rb") as f:
      return pickle.load(f)
  except Exception as e:
    return None


model = load_model()

# Aesthetic Banner with Topic-Related Image
st.markdown(
    """
    <div class="banner-container">
        <img src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=1200&q=80" class="banner-img">
        <div class="banner-text">
            <h1 class="banner-title">🎓 Student Performance Predictor</h1>
            <p class="banner-subtitle">Advanced Machine Learning Model Powered by K-Nearest Neighbors (KNN)</p>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

# Input Form inside Streamlit
with st.form("prediction_form"):
  st.markdown("### 📊 Enter Student Parameters")
  st.markdown(
      "Please fill in the details below to estimate the academic performance"
      " outcome."
  )

  col1, col2 = st.columns(2)

  with col1:
    age = st.number_input("Age", value=18.0, step=1.0)
    gender = st.number_input("Gender (Encoded/Numeric)", value=0.0, step=1.0)
    course = st.number_input("Course (Encoded/Numeric)", value=0.0, step=1.0)
    study_hours = st.number_input("Study Hours", value=5.0, step=0.5)
    class_attendance = st.number_input("Class Attendance (%)", value=85.0, step=1.0)
    internet_access = st.number_input(
        "Internet Access (0 or 1)", value=1.0, step=1.0
    )

  with col2:
    sleep_hours = st.number_input("Sleep Hours", value=7.0, step=0.5)
    sleep_quality = st.number_input("Sleep Quality (1-5)", value=3.0, step=1.0)
    study_method = st.number_input("Study Method (Encoded)", value=1.0, step=1.0)
    facility_rating = st.number_input(
        "Facility Rating (1-5)", value=4.0, step=1.0
    )
    exam_difficulty = st.number_input(
        "Exam Difficulty (1-5)", value=3.0, step=1.0
    )

  st.markdown("<br>", unsafe_allow_html=True)
  submitted = st.form_submit_button("🚀 Run Prediction")

if submitted:
  try:
    features = [
        age,
        gender,
        course,
        study_hours,
        class_attendance,
        internet_access,
        sleep_hours,
        sleep_quality,
        study_method,
        facility_rating,
        exam_difficulty,
    ]

    if model:
      input_data = np.array([features])
      pred = model.predict(input_data)
      prediction = round(float(pred[0]), 4)

      # Gorgeous Result Card
      st.markdown(
          f"""
            <div style="padding: 20px; background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 12px; text-align: center; margin-top: 20px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);">
                <span style="font-size: 12px; text-transform: uppercase; letter-spacing: 2px; color: #a5b4fc; font-weight: 700;">Predicted Outcome Result</span>
                <div style="font-size: 36px; font-weight: 900; color: #38bdf8; margin-top: 5px;">{prediction}</div>
            </div>
            """,
          unsafe_allow_html=True,
      )
    else:
      st.error(
          "⚠️ Model file (`knn.pkl`) could not be loaded. Please ensure it is"
          " uploaded to your repository."
      )
  except Exception as e:
    st.error(f"❌ Error in processing input: {e}")

# Footer
st.markdown(
    """
    <div style="text-align: center; margin-top: 40px; font-size: 12px; color: #64748b;">
        Crafted with Streamlit & Python • Designed for Cloud Deployment
    </div>
""",
    unsafe_allow_html=True,
)
