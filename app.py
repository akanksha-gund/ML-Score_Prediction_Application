import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title='K-NN Regressor Deployment', page_icon='📊', layout='centered'
)


# Load the trained model safely
@st.cache_resource
def load_model():
  with open('knn.pkl', 'rb') as file:
    model = pickle.load(file)
  return model


try:
  model = load_model()
except Exception as e:
  st.error(
      f'Error loading `knn.pkl`. Please ensure the file is in the correct'
      f' directory. Details: {e}'
  )
  st.stop()

st.title('🎓 Student Performance Prediction App')
st.write(
    'Provide the student details below to get a prediction using your trained'
    ' **KNeighborsRegressor** model.'
)

st.divider()

# Input Form Layout
with st.form('prediction_form'):
  st.subheader('Enter Student Parameters')

  col1, col2 = st.columns(2)

  with col1:
    age = st.number_input('Age[cite: 1]', min_value=10, max_value=100, value=20)
    gender = st.sidebar if False else st.selectbox(
        'Gender[cite: 1]', ['Male', 'Female', 'Other']
    )
    course = st.selectbox(
        'Course[cite: 1]', ['Science', 'Commerce', 'Arts', 'Engineering', 'Other']
    )
    study_hours = st.number_input(
        'Study Hours[cite: 1]', min_value=0.0, max_value=24.0, value=4.0
    )
    class_attendance = st.slider(
        'Class Attendance (%)[cite: 1]', min_value=0, max_value=100, value=75
    )
    internet_access = st.selectbox(
        'Internet Access[cite: 1]', ['Yes', 'No']
    )

  with col2:
    sleep_hours = st.number_input(
        'Sleep Hours[cite: 1]', min_value=0.0, max_value=16.0, value=7.0
    )
    sleep_quality = st.selectbox(
        'Sleep Quality[cite: 1]', ['Poor', 'Average', 'Good']
    )
    study_method = st.selectbox(
        'Study Method[cite: 1]',
        ['Self-Study', 'Group Study', 'Online', 'Offline Classes'],
    )
    facility_rating = st.slider(
        'Facility Rating (1-5)[cite: 1]', min_value=1, max_value=5, value=3
    )
    exam_difficulty = st.selectbox(
        'Exam Difficulty[cite: 1]', ['Easy', 'Medium', 'Hard']
    )

  submit_button = st.form_submit_button(
      label='Predict', use_container_width=True
)

if submit_button:
  # Map categorical features to numeric values
  # Note: Adjust these mappings if your training script used different numeric labels
  gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
  internet_map = {'No': 0, 'Yes': 1}
  sleep_quality_map = {'Poor': 0, 'Average': 1, 'Good': 2}
  exam_diff_map = {'Easy': 0, 'Medium': 1, 'Hard': 2}

  # Construct input array matching the model's feature layout[cite: 1]
  # Ensure the order of columns strictly matches your training dataset columns:
  # age, gender, course, study_hours, class_attendance, internet_access, sleep_hours, sleep_quality, study_method, facility_rating, exam_difficulty[cite: 1]

  # For demonstration, simple numeric placeholders/hashes are used for text-heavy columns like 'course' and 'study_method'
  # If your model used OneHotEncoding, you must transform the input accordingly.
  input_data = pd.DataFrame(
      [[
          age,
          gender_map.get(gender, 0),
          hash(course) % 10,
          study_hours,
          class_attendance,
          internet_map.get(internet_access, 0),
          sleep_hours,
          sleep_quality_map.get(sleep_quality, 1),
          hash(study_method) % 10,
          facility_rating,
          exam_diff_map.get(exam_difficulty, 1),
      ]],
      columns=[
          'age',
          'gender',
          'course',
          'study_hours',
          'class_attendance',
          'internet_access',
          'sleep_hours',
          'sleep_quality',
          'study_method',
          'facility_rating',
          'exam_difficulty',
      ],
  )

  try:
    prediction = model.predict(input_data)
    st.success(f'### Predicted Output: {prediction[0]:.2f}')
  except Exception as err:
    st.error(
        'Prediction failed. Please ensure your input data encoding matches'
        f' the format expected by the model. Details: {err}'
    )
