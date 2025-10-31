# ==========================================================
# 🧬 hERG Cardiotoxicity Predictor — Streamlit Web App
# Rebuilt to use PubChemPy instead of RDKit
# Works on Streamlit Cloud (no RDKit dependency)
# ==========================================================

# -------------------------------
# ✅ IMPORTS
# -------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

try:
    from pubchempy import get_compounds
    pubchem_available = True
except ImportError:
    pubchem_available = False

# -------------------------------
# ✅ STREAMLIT CONFIG (must be first)
# -------------------------------
st.set_page_config(
    page_title="🧬 hERG Cardiotoxicity Predictor",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------
# ✅ MODEL LOADING
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("herg_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Model or scaler file missing: {e}")
        st.stop()

# -------------------------------
# ✅ FEATURE GENERATION VIA PUBCHEMPY
# -------------------------------
def smiles_to_features(smiles):
    """Extract molecular descriptors from PubChem."""
    if not pubchem_available:
        st.error("❌ PubChemPy not installed.")
        return None

    try:
        compound = get_compounds(smiles, "smiles")[0]
    except Exception:
        return None

    try:
        features = {
            "MolWt": compound.molecular_weight,
            "LogP": compound.xlogp,
            "TPSA": compound.tpsa,
            "HBD": compound.hbond_donor_count,
            "HBA": compound.hbond_acceptor_count,
            "RotatableBonds": compound.rotatable_bond_count,
            "HeavyAtoms": compound.heavy_atom_count,
            "Complexity": compound.complexity,
            "Charge": compound.charge,
            "ExactMass": compound.exact_mass,
        }
        return features
    except Exception:
        return None

# -------------------------------
# ✅ LOAD MODEL + SCALER
# -------------------------------
model, scaler = load_model()

# -------------------------------
# ✅ PAGE CONTENT
# -------------------------------
st.title("🧬 hERG Cardiotoxicity Predictor (PubChem-based)")

st.markdown("""
This tool predicts whether a compound blocks the **hERG potassium ion channel** —  
a major cause of **cardiotoxicity** in drug discovery pipelines.

Enter a SMILES string below to get:
- Extracted molecular features from **PubChem**
- hERG Toxicity prediction (`Toxic ⚠️` or `Non-toxic ✅`)
- Confidence scores and feature visualization
""")

st.sidebar.header("ℹ️ About the Model")
st.sidebar.info("""
**Model:** Random Forest Classifier  
**Trained on:** TDC hERG Dataset  
**Feature Source:** PubChem via PubChemPy  
**Input:** SMILES notation  
**Output:** Toxicity Prediction
""")

# -------------------------------
# ✅ USER INPUT
# -------------------------------
col1, col2 = st.columns([2, 1])
with col1:
    smiles_input = st.text_input(
        "Enter SMILES string:",
        placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)"
    )

    predict_button = st.button("🔍 Predict Toxicity", type="primary")

# -------------------------------
# ✅ MAIN LOGIC
# -------------------------------
if predict_button:
    if not smiles_input.strip():
        st.warning("⚠️ Please enter a valid SMILES string.")
        st.stop()

    with st.spinner("Fetching molecular data from PubChem..."):
        features = smiles_to_features(smiles_input)

    if not features:
        st.error("❌ Could not fetch molecular data for this SMILES. Try another one.")
        st.stop()

    # Convert to dataframe for model input
    X = pd.DataFrame([features])
    try:
        X_scaled = scaler.transform(X)
    except Exception:
        X_scaled = X  # fallback if no scaling needed

    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]

    # -------------------------------
    # ✅ DISPLAY RESULTS
    # -------------------------------
    st.header("🧠 Prediction Results")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Prediction", "TOXIC ⚠️" if pred == 1 else "NON-TOXIC ✅")
    with col_b:
        st.metric("Confidence (Non-toxic)", f"{prob[0]*100:.2f}%")
    with col_c:
        st.metric("Confidence (Toxic)", f"{prob[1]*100:.2f}%")

    # -------------------------------
    # ✅ FEATURE TABLE
    # -------------------------------
    st.subheader("🔬 Extracted Molecular Features")
    feat_df = pd.DataFrame(features.items(), columns=["Descriptor", "Value"])
    st.dataframe(feat_df, use_container_width=True)

    # -------------------------------
    # ✅ BAR CHART
    # -------------------------------
    st.subheader("📊 Descriptor Overview")
    fig = px.bar(
        x=list(features.keys()),
        y=list(features.values()),
        labels={"x": "Descriptor", "y": "Value"},
        title="Molecular Descriptor Values"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # ✅ INTERPRETATION
    # -------------------------------
    if pred == 1:
        st.warning("""
        ⚠️ **Warning:** This compound is predicted to block the hERG channel.  
        Such molecules may cause **QT interval prolongation** and **cardiac arrhythmia**.
        """)
    else:
        st.success("""
        ✅ **Safe Prediction:** This compound is unlikely to block the hERG channel.  
        However, please note this is **only one aspect of drug safety evaluation.**
        """)

# -------------------------------
# ✅ FOOTER
# -------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center'>
    <p>🧠 Data Source: <a href='https://tdcommons.ai/'>Therapeutics Data Commons (TDC)</a></p>
    <p>⚗️ Features retrieved via <a href='https://pubchempy.readthedocs.io/'>PubChemPy</a></p>
    <p>Developed by <b>Your Name</b> | © 2025 Bioinformatics Project</p>
</div>
""", unsafe_allow_html=True)
