# ==========================================================
# 🧬 hERG Cardiotoxicity Predictor — Streamlit Web App
# Rebuilt to use PubChemPy instead of RDKit
# Works on Streamlit Cloud (no RDKit dependency)
# ==========================================================

import streamlit as st

# -------------------------------
# ✅ STREAMLIT CONFIG (MUST BE FIRST!)
# -------------------------------
st.set_page_config(
    page_title="🧬 hERG Cardiotoxicity Predictor",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------
# ✅ IMPORTS (after set_page_config)
# -------------------------------
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

try:
    from pubchempy import get_compounds
    pubchem_available = True
except ImportError:
    pubchem_available = False
    st.error("❌ PubChemPy not installed. Please add it to requirements.txt")

# -------------------------------
# ✅ MODEL LOADING
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("herg_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"❌ Model or scaler file missing: {e}")
        st.info("Please ensure 'herg_model.pkl' and 'scaler.pkl' are in the root directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
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
        compounds = get_compounds(smiles, "smiles")
        if not compounds:
            return None
        compound = compounds[0]
    except Exception as e:
        st.error(f"Error fetching compound: {e}")
        return None

    try:
        features = {
            "MolWt": compound.molecular_weight or 0,
            "LogP": compound.xlogp or 0,
            "TPSA": compound.tpsa or 0,
            "HBD": compound.hbond_donor_count or 0,
            "HBA": compound.hbond_acceptor_count or 0,
            "RotatableBonds": compound.rotatable_bond_count or 0,
            "HeavyAtoms": compound.heavy_atom_count or 0,
            "Complexity": compound.complexity or 0,
            "Charge": compound.charge or 0,
            "ExactMass": compound.exact_mass or 0,
        }
        return features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# -------------------------------
# ✅ LOAD MODEL + SCALER
# -------------------------------
try:
    model, scaler = load_model()
except:
    st.stop()

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

st.sidebar.header("📝 Example SMILES")
st.sidebar.code("CC(=O)OC1=CC=CC=C1C(=O)O", language="text")
st.sidebar.caption("Aspirin (Non-toxic)")
st.sidebar.code("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", language="text")
st.sidebar.caption("Caffeine")

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
        features = smiles_to_features(smiles_input.strip())

    if not features:
        st.error("❌ Could not fetch molecular data for this SMILES. Please check if it's valid and try again.")
        st.stop()

    # Convert to dataframe for model input
    X = pd.DataFrame([features])
    
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        st.warning(f"Scaling failed, using raw features: {e}")
        X_scaled = X.values

    try:
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0]
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # -------------------------------
    # ✅ DISPLAY RESULTS
    # -------------------------------
    st.header("🧠 Prediction Results")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if pred == 1:
            st.metric("Prediction", "TOXIC ⚠️", delta="High Risk", delta_color="inverse")
        else:
            st.metric("Prediction", "NON-TOXIC ✅", delta="Low Risk", delta_color="normal")
    with col_b:
        st.metric("Confidence (Non-toxic)", f"{prob[0]*100:.2f}%")
    with col_c:
        st.metric("Confidence (Toxic)", f"{prob[1]*100:.2f}%")

    # -------------------------------
    # ✅ FEATURE TABLE
    # -------------------------------
    st.subheader("🔬 Extracted Molecular Features")
    feat_df = pd.DataFrame(features.items(), columns=["Descriptor", "Value"])
    feat_df["Value"] = feat_df["Value"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # -------------------------------
    # ✅ BAR CHART
    # -------------------------------
    st.subheader("📊 Descriptor Overview")
    fig = px.bar(
        x=list(features.keys()),
        y=list(features.values()),
        labels={"x": "Descriptor", "y": "Value"},
        title="Molecular Descriptor Values",
        color=list(features.values()),
        color_continuous_scale="viridis"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # ✅ INTERPRETATION
    # -------------------------------
    st.subheader("🔍 Interpretation")
    if pred == 1:
        st.warning("""
        ⚠️ **Warning:** This compound is predicted to block the hERG channel.  
        Such molecules may cause **QT interval prolongation** and **cardiac arrhythmia**.
        
        **Recommended Actions:**
        - Further experimental validation needed
        - Consider structural modifications
        - Evaluate risk-benefit ratio
        """)
    else:
        st.success("""
        ✅ **Safe Prediction:** This compound is unlikely to block the hERG channel.  
        
        **Important Notes:**
        - This is only one aspect of drug safety evaluation
        - Further toxicity studies are still recommended
        - In vitro and in vivo validation required
        """)

# -------------------------------
# ✅ FOOTER
# -------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center'>
    <p>🧠 Data Source: <a href='https://tdcommons.ai/' target='_blank'>Therapeutics Data Commons (TDC)</a></p>
    <p>⚗️ Features retrieved via <a href='https://pubchempy.readthedocs.io/' target='_blank'>PubChemPy</a></p>
    <p>Developed for Bioinformatics Project | © 2025</p>
</div>
""", unsafe_allow_html=True)
