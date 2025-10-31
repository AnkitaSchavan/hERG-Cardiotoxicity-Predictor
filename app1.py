# ==========================================================
# 🧬 hERG Cardiotoxicity Predictor — Streamlit Web App
# Works both locally (with RDKit) and online (without RDKit)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px

# Try importing RDKit safely
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Draw
    rdkit_available = True
except ImportError:
    rdkit_available = False

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
    model = joblib.load("herg_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

# ----------------------------------------------------------
# FEATURE GENERATION FUNCTION
# ----------------------------------------------------------
def smiles_to_features(smiles):
    """Convert SMILES to molecular descriptors."""
    if not rdkit_available:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    features = {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "NumHeavyAtoms": mol.GetNumHeavyAtoms(),
        "NumRings": rdMolDescriptors.CalcNumRings(mol),
    }
    return features

# ----------------------------------------------------------
# HEADER & SIDEBAR
# ----------------------------------------------------------
st.title("🧬 hERG Cardiotoxicity Predictor")

st.markdown("""
This application predicts whether a molecule blocks the **hERG potassium channel**, 
a key factor in assessing cardiac toxicity risk during drug development.
""")

st.sidebar.header("About")
st.sidebar.info("""
**Model:** Random Forest Classifier  
**Dataset:** TDC hERG Toxicity  
**Input:** SMILES molecular structure  
**Output:** Toxic (1) or Non-toxic (0)
""")

st.sidebar.header("Example SMILES")
st.sidebar.code("CCO  # Ethanol")
st.sidebar.code("CC(=O)OC1=CC=CC=C1C(=O)O  # Aspirin")
st.sidebar.code("CN1C=NC2=C1C(=O)N(C(=O)N2C)C  # Caffeine")

# ----------------------------------------------------------
# MAIN CONTENT
# ----------------------------------------------------------
try:
    model, scaler = load_model()
    st.success("✅ Model loaded successfully!")

    st.header("Enter Molecule Information")

    col1, col2 = st.columns([2, 1])

    with col1:
        smiles_input = st.text_input(
            "Enter SMILES notation:",
            placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O",
            help="SMILES (Simplified Molecular Input Line Entry System) represents molecular structures."
        )

        if st.button("🔍 Predict Toxicity", type="primary"):
            if not smiles_input:
                st.warning("Please enter a SMILES notation.")
            elif not rdkit_available:
                st.error("❌ RDKit is not available. Molecular feature extraction cannot proceed.")
            else:
                with st.spinner("Analyzing molecule..."):
                    mol = Chem.MolFromSmiles(smiles_input)
                    if mol is None:
                        st.error("❌ Invalid SMILES notation. Please check your input.")
                    else:
                        features = smiles_to_features(smiles_input)
                        if features is None:
                            st.error("❌ Could not extract molecular features.")
                        else:
                            features_df = pd.DataFrame([features])
                            features_scaled = scaler.transform(features_df)

                            prediction = model.predict(features_scaled)[0]
                            prediction_proba = model.predict_proba(features_scaled)[0]

                            st.header("Prediction Results")

                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Prediction", "TOXIC ⚠️" if prediction == 1 else "NON-TOXIC ✅")
                            with col_b:
                                st.metric("Confidence (Non-toxic)", f"{prediction_proba[0]:.2%}")
                            with col_c:
                                st.metric("Confidence (Toxic)", f"{prediction_proba[1]:.2%}")

                            # ---------------------- Visualization ----------------------
                            st.subheader("Molecular Structure")
                            img = Draw.MolToImage(mol, size=(400, 400))
                            st.image(img, caption="2D Molecular Structure")

                            # Molecular Properties
                            st.subheader("Molecular Properties")
                            props_df = pd.DataFrame([features]).T
                            props_df.columns = ["Value"]
                            st.dataframe(props_df, use_container_width=True)

                            # Feature visualization
                            st.subheader("Feature Contribution Overview")
                            fig = px.bar(
                                x=list(features.keys()),
                                y=list(features.values()),
                                labels={"x": "Descriptor", "y": "Value"},
                                title="Molecular Descriptor Values"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Warning / success
                            if prediction == 1:
                                st.warning("""
                                ⚠️ **Warning:** This molecule is predicted to block the hERG channel, 
                                which may cause **cardiac arrhythmias** and other heart-related side effects.
                                """)
                            else:
                                st.success("""
                                ✅ **Good News:** This molecule is predicted to be safe regarding hERG toxicity.  
                                (Note: This is only one aspect of drug safety.)
                                """)

    with col2:
        st.info("""
        **How to use:**
        1. Enter a valid SMILES string  
        2. Click **Predict Toxicity**  
        3. Review the toxicity prediction and confidence levels

        **Note:** RDKit is required for molecular feature extraction.
        """)

except FileNotFoundError:
    st.error("""
    ❌ Model files not found!  
    Please ensure `herg_model.pkl` and `scaler.pkl` exist in the working directory.
    """)

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🧠 Data Source: <a href='https://tdcommons.ai/'>Therapeutics Data Commons (TDC)</a></p>
    <p>⚗️ Model trained on real hERG toxicity data using Random Forest</p>
</div>
""", unsafe_allow_html=True)
