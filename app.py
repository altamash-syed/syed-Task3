import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib


# Set page configuration
st.set_page_config(
    page_title="Loan Default Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 50px;
    }
    .stButton>button:hover {
        background-color: #004999;
        color: white;
    }
    .css-1v0mbdj {
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'loan_model.pkl')
        model = joblib.load(model_path)
        return model
    except Exception as e:
        return None

model = load_model()

st.title("💰 Loan Risk & Approval Prediction Dashboard")
st.markdown("Provide the applicant details below to evaluate risk. Categorical variables are simplified into intuitive dropdown selections.")

with st.form("prediction_form"):
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Loan & Financials")
        loan_amnt = st.number_input("Loan Amount ($)", min_value=500.0, max_value=40000.0, value=15000.0, step=500.0)
        funded_amnt = st.number_input("Funded Amount ($)", min_value=500.0, max_value=40000.0, value=15000.0, step=500.0)
        
        term_options = {" 36 months": 36, " 60 months": 60}
        term_label = st.selectbox("Term", options=list(term_options.keys()))
        term = term_options[term_label] # If model expects numeric or specific string format, adjust accordingly
        
        int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=30.0, value=11.5, step=0.1)
        installment = st.number_input("Monthly Installment ($)", min_value=10.0, max_value=1500.0, value=350.0, step=10.0)
        annual_inc = st.number_input("Annual Income ($)", min_value=1000.0, max_value=2000000.0, value=75000.0, step=1000.0)

    with col2:
        st.subheader("Credit Profile & History")
        # Sub grade mapping or selection
        sub_grades = [
            'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5',
            'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5',
            'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5'
        ]
        sub_grade = st.selectbox("Sub Grade", options=sub_grades, index=5)
        
        emp_length_options = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
            '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10
        }
        emp_length = st.selectbox("Employment Length", options=list(emp_length_options.keys()))
        
        dti = st.slider("Debt-To-Income Ratio (DTI)", min_value=0.0, max_value=100.0, value=18.5, step=0.1)
        delinq_2yrs = st.number_input("Delinquencies (Past 2 Years)", min_value=0, max_value=30, value=0)
        
        # Earliest Credit Line (Can be year or date representation depending on preprocessing, standardizing as year integer or string)
        earliest_cr_line = st.number_input("Earliest Credit Line (Year)", min_value=1950, max_value=2025, value=2005)
        inq_last_6mths = st.number_input("Inquiries in Last 6 Months", min_value=0, max_value=20, value=0)

    with col3:
        st.subheader("Accounts & Balances")
        open_acc = st.number_input("Open Credit Accounts", min_value=0, max_value=100, value=8)
        pub_rec = st.number_input("Public Records", min_value=0, max_value=20, value=0)
        revol_bal = st.number_input("Revolving Balance ($)", min_value=0.0, max_value=200000.0, value=12000.0, step=100.0)
        revol_util = st.slider("Revolving Line Utilization Rate (%)", min_value=0.0, max_value=150.0, value=45.0, step=0.5)
        total_acc = st.number_input("Total Credit Accounts", min_value=1, max_value=150, value=20)
        collections_12_mths_ex_med = st.number_input("Collections (12 mos excluding medical)", min_value=0, max_value=20, value=0)

    st.markdown("---")
    st.subheader("Additional Financial & Categorical Attributes")
    col4, col5, col6 = st.columns(3)

    with col4:
        acc_now_delinq = st.number_input("Accounts Now Delinquent", min_value=0, max_value=10, value=0)
        tot_coll_amt = st.number_input("Total Collection Amount Ever ($)", min_value=0.0, max_value=100000.0, value=0.0, step=100.0)

    with col5:
        tot_cur_bal = st.number_input("Total Current Balance ($)", min_value=0.0, max_value=5000000.0, value=140000.0, step=1000.0)
        total_rev_hi_lim = st.number_input("Total Revolving High Limit ($)", min_value=0.0, max_value=2000000.0, value=35000.0, step=500.0)

    with col6:
        # Dropdown for Home Ownership (consolidating OTHER, OWN, RENT, MORTGAGE if applicable)
        home_ownership_choices = ['RENT', 'OWN', 'OTHER', 'MORTGAGE']
        home_ownership = st.selectbox("Home Ownership", options=home_ownership_choices)

        # Dropdown for Purpose
        purpose_choices = [
            'credit_card', 'debt_consolidation', 'educational', 'home_improvement',
            'house', 'major_purchase', 'medical', 'moving', 'other',
            'renewable_energy', 'small_business', 'vacation', 'wedding'
        ]
        purpose = st.selectbox("Loan Purpose", options=purpose_choices)

        # Dropdown for Verification Status
        verification_status_choices = ['Not Verified', 'Source Verified', 'Verified']
        verification_status = st.selectbox("Verification Status", options=verification_status_choices)

    submitted = st.form_submit_button("Predict Loan Status")

if submitted:
    # Build a base dictionary with numeric inputs matching the model's exact expected features
    input_data = {
        'loan_amnt': loan_amnt,
        'funded_amnt': funded_amnt,
        'term': term, # Ensure format matches training data (e.g. integer 36/60 or string)
        'int_rate': int_rate,
        'installment': installment,
        'sub_grade': sub_grade, # Note: if sub_grade was ordinal/encoded during training, map accordingly or pass directly if model handles categorical pipelines
        'emp_length': emp_length_options[emp_length],
        'annual_inc': annual_inc,
        'dti': dti,
        'delinq_2yrs': delinq_2yrs,
        'earliest_cr_line': earliest_cr_line,
        'inq_last_6mths': inq_last_6mths,
        'open_acc': open_acc,
        'pub_rec': pub_rec,
        'revol_bal': revol_bal,
        'revol_util': revol_util,
        'total_acc': total_acc,
        'collections_12_mths_ex_med': collections_12_mths_ex_med,
        'acc_now_delinq': acc_now_delinq,
        'tot_coll_amt': tot_coll_amt,
        'tot_cur_bal': tot_cur_bal,
        'total_rev_hi_lim': total_rev_hi_lim,
        
        # One-Hot Encoded columns for Home Ownership
        'home_ownership_OTHER': 1 if home_ownership == 'OTHER' else 0,
        'home_ownership_OWN': 1 if home_ownership == 'OWN' else 0,
        'home_ownership_RENT': 1 if home_ownership == 'RENT' else 0,
        
        # One-Hot Encoded columns for Purpose
        'purpose_credit_card': 1 if purpose == 'credit_card' else 0,
        'purpose_debt_consolidation': 1 if purpose == 'debt_consolidation' else 0,
        'purpose_educational': 1 if purpose == 'educational' else 0,
        'purpose_home_improvement': 1 if purpose == 'home_improvement' else 0,
        'purpose_house': 1 if purpose == 'house' else 0,
        'purpose_major_purchase': 1 if purpose == 'major_purchase' else 0,
        'purpose_medical': 1 if purpose == 'medical' else 0,
        'purpose_moving': 1 if purpose == 'moving' else 0,
        'purpose_other': 1 if purpose == 'other' else 0,
        'purpose_renewable_energy': 1 if purpose == 'renewable_energy' else 0,
        'purpose_small_business': 1 if purpose == 'small_business' else 0,
        'purpose_vacation': 1 if purpose == 'vacation' else 0,
        'purpose_wedding': 1 if purpose == 'wedding' else 0,
        
        # One-Hot Encoded columns for Verification Status
        'verification_status_Source Verified': 1 if verification_status == 'Source Verified' else 0,
        'verification_status_Verified': 1 if verification_status == 'Verified' else 0
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    if model is not None:
        try:
            # Ensure column order matches the exact feature list provided
            expected_columns = [
                'loan_amnt', 'funded_amnt', 'term', 'int_rate', 'installment', 'sub_grade', 
                'emp_length', 'annual_inc', 'dti', 'delinq_2yrs', 'earliest_cr_line', 
                'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 
                'total_acc', 'collections_12_mths_ex_med', 'acc_now_delinq', 'tot_coll_amt', 
                'tot_cur_bal', 'total_rev_hi_lim', 'home_ownership_OTHER', 'home_ownership_OWN', 
                'home_ownership_RENT', 'purpose_credit_card', 'purpose_debt_consolidation', 
                'purpose_educational', 'purpose_home_improvement', 'purpose_house', 
                'purpose_major_purchase', 'purpose_medical', 'purpose_moving', 'purpose_other', 
                'purpose_renewable_energy', 'purpose_small_business', 'purpose_vacation', 
                'purpose_wedding', 'verification_status_Source Verified', 'verification_status_Verified'
            ]
            
            # Handle potential sub_grade encoding if model expects numeric label encoding instead of string
            # If your joblib model pipeline handles string categorical encoding, it will process 'sub_grade' automatically.
            
            # Reindex dataframe to match columns order if needed
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[expected_columns]

            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df) if hasattr(model, "predict_proba") else None

            st.markdown("---")
            st.subheader("Prediction Result")
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                if prediction[0] == 1 or str(prediction[0]).lower() in ['bad', 'default', 'high risk']:
                    st.error("⚠️️ **High Risk Loan / Likely to Default**")
                else:
                    st.success("✅ **Low Risk Loan / Good Standing Candidate**")
            
            with col_res2:
                if prediction_proba is not None:
                    confidence = np.max(prediction_proba) * 100
                    st.metric(label="Model Confidence Score", value=f"{confidence:.2f}%")

            with st.expander("View Submitted Feature Vector"):
                st.dataframe(input_df)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("Tip: Ensure your loaded `joblib` model matches the exact feature columns structure.")
    else:
        st.warning("Model file (`loan_model.pkl`) not found in directory. Please upload your trained joblib model to enable live predictions.")