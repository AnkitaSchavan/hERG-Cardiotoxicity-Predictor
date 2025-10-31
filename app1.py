# ==========================================================
# 🧬 hERG Cardiotoxicity Predictor — Streamlit Web App
# Works seamlessly on Streamlit Cloud (no RDKit dependency)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pubchempy import get_compounds

# ✅ PAGE CONFIG — must be the FIRST Streamlit command
st.set_page_config(
    page_title="🧬 hERG Cardiotoxicity Predictor",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    """Load pretrained hERG model and scaler"""
    model = joblib.load("herg_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names


try:
    model, scaler, feature_names = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model/scaler files: {e}")
    st.stop()

# ----------------------------------------------------------
# PUBCHEMPY FEATURE EXTRACTION
# ----------------------------------------------------------
def get_pubchem_features(compound_name):
    """Fetch basic molecular properties from PubChem using PubChemPy"""
    try:
        compound = get_compounds(compound_name, 'name')[0]
        features = {
            "MolecularWeight": compound.molecular_weight,
            "XLogP": compound.xlogp,
            "HBondDonorCount": compound.hbond_donor_count,
            "HBondAcceptorCount": compound.hbond_acceptor_count,
            "RotatableBondCount": compound.rotatable_bond_count,
            "TPSA": compound.tpsa,
        }
        image_url = compound.image  # 2D image from PubChem
        return features, image_url
    except Exception as e:
        st.error(f"Could not retrieve PubChem data for '{compound_name}'. Error: {e}")
        return None, None

# ----------------------------------------------------------
# SIDEBAR INFO
# ----------------------------------------------------------
st.sidebar.header("About")
st.sidebar.info("""
**Model:** Random Forest Classifier  
**Dataset:** TDC hERG Toxicity  
**Input:** Compound name (PubChem)  
**Output:** Toxic (1) or Non-toxic (0)
""")

st.sidebar.header("Example Compounds")
st.sidebar.code("aspirin")
st.sidebar.code("caffeine")
st.sidebar.code("amiodarone")
st.sidebar.code("ibuprofen")

# ----------------------------------------------------------
# MAIN PAGE CONTENT
# ----------------------------------------------------------
st.title("🧬 hERG Cardiotoxicity Predictor (PubChem Edition)")

st.markdown("""
Predict whether a given compound may block the **hERG potassium channel**, 
a critical marker of **cardiac toxicity** in drug discovery.

This version automatically retrieves compound properties from **PubChem**.
""")

col1, col2 = st.columns([2, 1])

with col1:
    compound_input = st.text_input(
        "Enter compound name or PubChem CID:",
        placeholder="e.g., aspirin, caffeine, 2244 (CID)",
        help="Enter the compound name or PubChem Compound ID (CID)"
    )

    if st.button("🔍 Predict Toxicity", type="primary"):
        if not compound_input:
            st.warning("⚠️ Please enter a compound name or CID.")
        else:
            with st.spinner("Fetching molecular properties from PubChem..."):
                features, image_url = get_pubchem_features(compound_input)

                if features:
                    st.subheader("🔬 Retrieved Molecular Features")
                    st.dataframe(pd.DataFrame([features]))

                    # Map PubChem features to model features
                    feature_vector = np.zeros(len(feature_names))
                    for i, f in enumerate(feature_names):
                        if f in features and features[f] is not None:
                            feature_vector[i] = features[f]

                    # Preprocess and predict
                    features_scaled = scaler.transform([feature_vector])
                    prediction = model.predict(features_scaled)[0]
                    prediction_proba = model.predict_proba(features_scaled)[0]

                    # ---------------------- Results ----------------------
                    st.header("Prediction Results")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Prediction", "TOXIC ⚠️" if prediction == 1 else "NON-TOXIC ✅")
                    with col_b:
                        st.metric("Confidence (Non-toxic)", f"{prediction_proba[0]:.2%}")
                    with col_c:
                        st.metric("Confidence (Toxic)", f"{prediction_proba[1]:.2%}")

                    # ---------------------- Visualization ----------------------
                    if image_url:
                        st.subheader("🧪 Molecular Structure")
                        st.image(image_url, caption="2D structure from PubChem", use_container_width=True)

                    st.subheader("📊 Molecular Descriptor Values")
                    fig = px.bar(
                        x=list(features.keys()),
                        y=list(features.values()),
                        labels={"x": "Descriptor", "y": "Value"},
                        title="Molecular Descriptor Overview"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # ---------------------- Interpretation ----------------------
                    if prediction == 1:
                        st.warning("""
                        ⚠️ **Warning:** This molecule is predicted to block the hERG channel, 
                        which may cause **cardiac arrhythmias** and other side effects.
                        """)
                    else:
                        st.success("""
                        ✅ **Safe:** This molecule is predicted to have a low risk of hERG toxicity.
                        (Note: Always consider multiple toxicity endpoints.)
                        """)

with col2:
    st.info("""
    ### How to Use
    1. Enter a valid compound name (e.g., aspirin)
    2. Click **Predict Toxicity**
    3. View molecular features, image, and prediction results

    ### Notes
    - Uses **PubChemPy** to fetch molecular properties
    - Works fully online (no RDKit needed)
    """)

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🧠 Data Source: <a href='https://tdcommons.ai/'>Therapeutics Data Commons (TDC)</a></p>
    <p>⚗️ Features from <a href='https://pubchem.ncbi.nlm.nih.gov/'>PubChem</a></p>
    <p>💡 Machine Learning Model: Random Forest</p>
</div>
""", unsafe_allow_html=True)
