import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model/final_model_weights.pkl")


# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(page_title="NTI ML Summer Training 2026", layout="wide")

st.markdown("""
<style>
    /* Enforce dark slide background */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    /* Left accent line container */
    .slide-container {
        border-left: 6px solid #00FFCC;
        padding-left: 2rem;
        margin-top: 2rem;
        margin-bottom: 3rem;
    }
    
    .nti-header {
        color: #00FFCC;
        font-size: 1.2rem;
        font-weight: 500;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    
    .main-title {
        font-size: 4.5rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0;
        color: #FFFFFF;
    }
    
    .highlight-title {
        font-size: 4.5rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0;
        color: #00FFCC;
    }
    
    .sub-title {
        font-size: 1.5rem;
        color: #94A3B8;
        margin-top: 1.5rem;
    }
    
    .sub-title strong {
        color: #FFFFFF;
    }
    
    /* Customizing Streamlit Inputs to fit the dark theme (Cards / Panels color) */
    div[data-baseweb="select"] > div, input[class^="st-"] {
        background-color: #161F2C !important;
        color: #FFFFFF !important;
        border-color: #94A3B8 !important;
    }
    
    /* Custom Prediction Result Box */
    .prediction-box-high {
        background-color: rgba(0, 255, 204, 0.1);
        border: 2px solid #00FFCC;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }
    .prediction-box-low {
        background-color: #161F2C;
        border: 2px solid #94A3B8;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }
    .pred-text {
        font-size: 2rem;
        font-weight: 800;
    }
    .pred-high { color: #00FFCC; }
    .pred-low { color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)


# --- 2. HEADER SECTION ---
st.markdown("""
<div class="slide-container">
    <div class="nti-header">National Telecommunication Institute (NTI)</div>
    <div class="main-title">Machine Learning</div>
    <div class="highlight-title">Summer Training 2026</div>
    <div class="sub-title">Implementation & Analysis of the <strong>Adult Income Dataset</strong></div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- 3. UI INPUT FORM ---
st.subheader("Predict Income Level (>50K or <=50K)")

with st.form("income_prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=17, max_value=90, value=35)
        ed_num = st.slider("Education Level (Years)", min_value=1, max_value=16, value=10)
        hours = st.slider("Hours per Week", min_value=1, max_value=99, value=40)
        
    with col2:
        cap_gain = st.number_input("Capital Gain ($)", min_value=0, value=0)
        cap_loss = st.number_input("Capital Loss ($)", min_value=0, value=0)
        sex = st.selectbox("Sex", ["Male", "Female"])
        race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
        
    with col3:
        workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "State/Local-gov", "Unknown", "infrequent_sklearn"])
        marital = st.selectbox("Marital Status", ["Married", "Never-married", "Divorced", "Separated", "Widowed", "Married-spouse-absent"])
        occupation = st.selectbox("Occupation", ["Prof-specialty", "Craft-repair", "Exec-managerial", "Adm-clerical", "Sales", "Other-service", "Machine-op-inspct", "Transport-moving", "Handlers-cleaners", "Farming-fishing", "Tech-support", "Protective-serv", "Priv-house-serv", "Unknown", "infrequent_sklearn"])
        relationship = st.selectbox("Relationship", ["Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"])
        country = st.selectbox("Native Country", ["United-States", "Mexico", "Philippines", "Germany", "Canada", "Other", "infrequent_sklearn"])

    submit_button = st.form_submit_button(label="Generate Prediction")

# --- 4. PREDICTION LOGIC & DATA PROCESSING ---
if submit_button:
    # Model expects 52 columns exactly matching training output
    cat_columns = [
        'cat__workclass_Federal-gov', 'cat__workclass_Private', 'cat__workclass_Self-emp-inc', 'cat__workclass_Self-emp-not-inc', 'cat__workclass_State/Local-gov', 'cat__workclass_Unknown', 'cat__workclass_infrequent_sklearn',
        'cat__marital.status_Divorced', 'cat__marital.status_Married', 'cat__marital.status_Married-spouse-absent', 'cat__marital.status_Never-married', 'cat__marital.status_Separated', 'cat__marital.status_Widowed',
        'cat__occupation_Adm-clerical', 'cat__occupation_Craft-repair', 'cat__occupation_Exec-managerial', 'cat__occupation_Farming-fishing', 'cat__occupation_Handlers-cleaners', 'cat__occupation_Machine-op-inspct', 'cat__occupation_Other-service', 'cat__occupation_Priv-house-serv', 'cat__occupation_Prof-specialty', 'cat__occupation_Protective-serv', 'cat__occupation_Sales', 'cat__occupation_Tech-support', 'cat__occupation_Transport-moving', 'cat__occupation_Unknown', 'cat__occupation_infrequent_sklearn',
        'cat__relationship_Not-in-family', 'cat__relationship_Other-relative', 'cat__relationship_Own-child', 'cat__relationship_Spouse', 'cat__relationship_Unmarried',
        'cat__race_Amer-Indian-Eskimo', 'cat__race_Asian-Pac-Islander', 'cat__race_Black', 'cat__race_Other', 'cat__race_White',
        'cat__sex_Female', 'cat__sex_Male',
        'cat__native.country_Canada', 'cat__native.country_Germany', 'cat__native.country_Mexico', 'cat__native.country_Other', 'cat__native.country_Philippines', 'cat__native.country_United-States', 'cat__native.country_infrequent_sklearn'
    ]
    
    # Initialize all features
    features = {
        'num__age': float(age),
        'num__education.num': float(ed_num),
        'num__capital.gain': float(cap_gain),
        'num__capital.loss': float(cap_loss),
        'num__hours.per.week': float(hours)
    }
    
    # Set default 0.0 for categorical columns
    for col in cat_columns:
        features[col] = 0.0

    # Activate Selected Categories to 1.0 (One-Hot Formatting)
    def activate_feature(prefix, val):
        col_name = f"cat__{prefix}_{val}"
        # Adjust 'Husband'/'Wife' to 'Spouse' based on relationship mapping from the pipeline if applicable,
        if val in ["Husband", "Wife"]: 
            col_name = "cat__relationship_Spouse"
            
        if col_name in features:
            features[col_name] = 1.0

    activate_feature("workclass", workclass)
    activate_feature("marital.status", marital)
    activate_feature("occupation", occupation)
    activate_feature("relationship", relationship)
    activate_feature("race", race)
    activate_feature("sex", sex)
    activate_feature("native.country", country)

    # Reorder columns explicitly to match model requirement
    expected_columns = ['num__age', 'num__education.num', 'num__capital.gain', 'num__capital.loss', 'num__hours.per.week'] + cat_columns
    
    # Build Input DataFrame
    df_input = pd.DataFrame([features], columns=expected_columns)
    
    # Run Prediction
    prediction = model.predict(df_input)[0]
    
    # --- 5. RESULT DISPLAY ---
    if prediction == 1:
        st.markdown("""
        <div class="prediction-box-high">
            <div style="color: #94A3B8; font-weight: 600; text-transform: uppercase;">Model Prediction</div>
            <div class="pred-text pred-high">Income > $50K</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="prediction-box-low">
            <div style="color: #94A3B8; font-weight: 600; text-transform: uppercase;">Model Prediction</div>
            <div class="pred-text pred-low">Income <= $50K</div>
        </div>
        """, unsafe_allow_html=True)