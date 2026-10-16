import pickle
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Student Performance Predictor", page_icon="🎓", layout="centered"
)

# Custom CSS for aesthetic design, blur effect banner, and dropdown styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    .banner-box {
        position: relative;
        width: 100%;
        height: 220px;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 15px 30px rgba(0,0,0,0.4);
        margin-bottom: 25px;
    }
    .banner-box img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        filter: brightness(0.6) blur(1px);
    }
    .banner-content {
        position: absolute;
        bottom: 20px;
        left: 25px;
        right: 25px;
        z-index: 2;
    }
    .banner-title {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        text-shadow: 0 2px 8px rgba(0,0,0,0.7);
    }
    .banner-sub {
        font-size: 13px;
        color: #cbd5e1;
        margin-top: 5px;
        text-shadow: 0 1px 4px rgba(0,0,0,0.7);
    }
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

# Load trained model
@st.cache_resource
def load_model():
  try:
    with open("knn.pkl", "rb") as f:
      return pickle.load(f)
  except:
    return None


model = load_model()

# Banner with title on top and blur effect
st.markdown(
    """
    <div class="banner-box">
        <img src="https://images.unsplash.com/photo-1523240795612-9a054b0db644?auto=format&fit=crop&w=1200&q=80">
        <div class="banner-content">
            <h1 class="banner-title">🎓 Student Performance Predictor</h1>
            <p class="banner-sub">Advanced Machine Learning Model Powered by K-Nearest Neighbors (KNN)</p>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
  st.markdown("### 📊 Enter Student Parameters")

  col1, col2 = st.columns(2)

  with col1:
    age = st.number_input("Age", value=18.0, step=1.0, min_value=10.0)

    # Select boxes instead of numeric inputs
    gender_sel = st.selectbox("Gender", ["Female", "Male"])
    gender = 0.0 if gender_sel == "Female" else 1.0

    course_sel = st.selectbox(
        "Course", ["Engineering", "Science", "Arts", "Commerce", "Other"]
    )
    course_map = {
        "Engineering": 0.0,
        "Science": 1.0,
        "Arts": 2.0,
        "Commerce": 3.0,
        "Other": 4.0,
    }
    course = course_map[course_sel]

    study_hours = st.number_input("Study Hours", value=5.0, step=0.5)
    class_attendance = st.number_input(
        "Class Attendance (%)", value=85.0, step=1.0
    )

    internet_sel = st.selectbox("Internet Access", ["Yes", "No"])
    internet_access = 1.0 if internet_sel == "Yes" else 0.0

  with col2:
    sleep_hours = st.number_input("Sleep Hours", value=7.0, step=0.5)
    sleep_quality = st.slider("Sleep Quality (1-5)", 1, 5, 3)
    facility_rating = st.slider("Facility Rating (1-5)", 1, 5, 4)
    exam_difficulty = st.slider("Exam Difficulty (1-5)", 1, 5, 3)

    study_method_sel = st.selectbox(
        "Study Method", ["Self Study", "Online", "Group", "Coaching"]
    )
    method_map = {
        "Self Study": 0.0,
        "Online": 1.0,
        "Group": 2.0,
        "Coaching": 3.0,
    }
    study_method = method_map[study_method_sel]

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
        float(sleep_quality),
        study_method,
        float(facility_rating),
        float(exam_difficulty),
    ]

    if model:
      pred = model.predict(np.array([features]))
      prediction = round(float(pred[0]), 4)
      st.markdown(
          f"""
            <div style="padding: 20px; background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 12px; text-align: center; margin-top: 20px;">
                <span style="font-size: 12px; text-transform: uppercase; letter-spacing: 2px; color: #a5b4fc; font-weight: 700;">Predicted Outcome Result</span>
                <div style="font-size: 36px; font-weight: 900; color: #38bdf8; margin-top: 5px;">{prediction}</div>
            </div>
            """,
          unsafe_allow_html=True,
      )
    else:
      st.error(
          "⚠️ Model file (`knn.pkl`) not found. Please upload it to your"
          " repository."
      )
  except Exception as e:
    st.error(f"❌ Error in processing input: {e}")
