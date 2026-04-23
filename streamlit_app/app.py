import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎓",
    layout="wide",
)

# INTEGRATED THEME CONFIGURATION (CSS)
st.markdown("""
    <style>
    /* Primary theme colors */
    :root {
        --primary-color: #1f77b4;
        --background-color: #ffffff;
        --secondary-background-color: #f0f2f6;
        --text-color: #262730;
    }
    
    /* Main container background */
    body, .stApp {
        background-color: #ffffff;
        color: #262730;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #262730 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1f77b4 !important;
        color: white !important;
        border-radius: 6px;
    }
    .stButton > button:hover {
        background-color: #1a5f99 !important;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border: 1px solid #1f77b4 !important;
        border-radius: 4px;
    }
    
    /* Metric container styling */
    [data-testid="stMetricContainer"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 15px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button {
        color: #262730 !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom-color: #1f77b4 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Student Placement Predictor")
st.write("Simple ML-based prediction system for student placement and salary")

#MODEL LOADING
@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        models_path = Path(__file__).parent.parent / "models"
        
        with open(models_path / "classification_best_model.pkl", 'rb') as f:
            clf_model = pickle.load(f)
        
        with open(models_path / "regression_best_model.pkl", 'rb') as f:
            reg_model = pickle.load(f)
        
        return clf_model, reg_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

#DATA PREPARATION
def prepare_features(data):
    try:
        df = pd.DataFrame([data])
        
        # Encode categorical columns
        categorical_mapping = {
            'gender': {'Female': 0, 'Male': 1},
            'branch': {'CSE': 0, 'IT': 1, 'ECE': 2, 'ME': 3, 'CE': 4},
            'part_time_job': {'No': 0, 'Yes': 1},
            'family_income_level': {'Low': 0, 'Middle': 1, 'High': 2},
            'city_tier': {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2},
            'internet_access': {'No': 0, 'Yes': 1},
            'extracurricular_involvement': {'No': 0, 'Yes': 1, 'Moderate': 2, 'Intensive': 3}
        }
        
        df_encoded = df.copy()
        for col, mapping in categorical_mapping.items():
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].map(mapping)
        
        # Define feature order EXACTLY as in training data A.csv
        feature_order = [
            'gender',
            'branch',
            'cgpa',
            'tenth_percentage',
            'twelfth_percentage',
            'backlogs',
            'study_hours_per_day',
            'attendance_percentage',
            'projects_completed',
            'internships_completed',
            'coding_skill_rating',
            'communication_skill_rating',
            'aptitude_skill_rating',
            'hackathons_participated',
            'certifications_count',
            'sleep_hours',
            'stress_level',
            'part_time_job',
            'family_income_level',
            'city_tier',
            'internet_access',
            'extracurricular_involvement'
        ]
        
        # Reorder columns to match exact training order
        df_final = df_encoded[feature_order]
        
        return df_final
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return None

#PREDICTIONS

def predict(clf_model, reg_model, student_data):
    """Make placement and salary predictions"""
    encoded_data = prepare_features(student_data)
    
    if encoded_data is None:
        return None, None
    
    try:
        # Placement prediction
        clf_pred = clf_model.predict(encoded_data)[0]
        clf_probs = clf_model.predict_proba(encoded_data)[0]
        
        placement = {
            'status': 'Placed' if clf_pred == 1 else 'Not Placed',
            'confidence': float(max(clf_probs) * 100),
            'probability': float(clf_probs[1] * 100 if len(clf_probs) > 1 else 0)
        }
        
        # Salary prediction
        salary = float(reg_model.predict(encoded_data)[0])
        
        if salary < 3:
            salary_range = "Below 3 LPA"
        elif salary < 5:
            salary_range = "3-5 LPA"
        elif salary < 7:
            salary_range = "5-7 LPA"
        else:
            salary_range = "7+ LPA"
        
        salary_pred = {
            'amount': salary,
            'range': salary_range
        }
        
        return placement, salary_pred
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

#MAIN APP

# Load models
clf_model, reg_model = load_models()

if clf_model is None or reg_model is None:
    st.stop()

#PREDICTION SECTION

st.subheader("Student Profile")

col1, col2, col3 = st.columns(3)

with col1:
    cgpa = st.number_input("CGPA (0-10)", 0.0, 10.0, 7.5, 0.1)
    tenth = st.number_input("10th %", 0.0, 100.0, 75.0, 1.0)
    twelfth = st.number_input("12th %", 0.0, 100.0, 78.0, 1.0)
    attendance = st.number_input("Attendance %", 0.0, 100.0, 85.0, 1.0)

with col2:
    coding = st.slider("Coding (1-5)", 1, 5, 3)
    communication = st.slider("Communication (1-5)", 1, 5, 3)
    aptitude = st.slider("Aptitude (1-5)", 1, 5, 3)
    projects = st.number_input("Projects", 0, 50, 3)

with col3:
    internships = st.number_input("Internships", 0, 10, 1)
    gender = st.selectbox("Gender", ["Female", "Male"])
    branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "ME", "CE"])
    backlogs = st.number_input("Backlogs", 0, 20, 0)

col4, col5, col6 = st.columns(3)

with col4:
    part_time = st.selectbox("Part-time Job", ["No", "Yes"])
    family_income = st.selectbox("Family Income", ["Low", "Middle", "High"])

with col5:
    city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    internet = st.selectbox("Internet Access", ["No", "Yes"])

with col6:
    extracurr = st.selectbox("Extracurricular", ["No", "Yes", "Moderate", "Intensive"])
    hackathons = st.number_input("Hackathons", 0, 20, 2)

# Additional inputs
col7, col8, col9 = st.columns(3)

with col7:
    study_hours = st.number_input("Study Hours/Day", 0.0, 24.0, 4.0, 0.5)
with col8:
    sleep_hours = st.number_input("Sleep Hours/Day", 0.0, 24.0, 7.0, 0.5)
with col9:
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)

# Predict button
if st.button("🔮 Predict", use_container_width=True):
    student_data = {
        'cgpa': cgpa,
        'tenth_percentage': tenth,
        'twelfth_percentage': twelfth,
        'attendance_percentage': attendance,
        'coding_skill_rating': coding,
        'communication_skill_rating': communication,
        'aptitude_skill_rating': aptitude,
        'projects_completed': projects,
        'internships_completed': internships,
        'gender': gender,
        'branch': branch,
        'backlogs': backlogs,
        'part_time_job': part_time,
        'family_income_level': family_income,
        'city_tier': city_tier,
        'internet_access': internet,
        'extracurricular_involvement': extracurr,
        'hackathons_participated': hackathons,
        'study_hours_per_day': study_hours,
        'sleep_hours': sleep_hours,
        'stress_level': stress,
        'certifications_count': 1
    }
    
    placement, salary = predict(clf_model, reg_model, student_data)
    
    if placement and salary:
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Placement Status")
            if placement['status'] == 'Placed':
                st.success(f"{placement['status']}")
            else:
                st.warning(f"{placement['status']}")
            
            st.metric("Confidence", f"{placement['confidence']:.1f}%")
            st.metric("Probability Placed", f"{placement['probability']:.1f}%")
        
        with col2:
            st.subheader("Salary Prediction")
            st.success(f"Rp {salary['amount']:.2f} LPA")
            st.metric("Range", salary['range'])


# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>Placement Prediction System | Built with Streamlit & scikit-learn</p>
</div>
""", unsafe_allow_html=True)
