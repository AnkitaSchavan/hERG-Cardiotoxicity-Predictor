# ==========================================================
# 🧬 hERG Cardiotoxicity Predictor
# ==========================================================

import streamlit as st

# THIS MUST BE THE ABSOLUTE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="🧬 hERG Cardiotoxicity Predictor",
    page_icon="🧠",
    layout="wide"
)

# Now import everything else
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

try:
    from pubchempy import get_compounds
    PUBCHEM_OK = True
except:
    PUBCHEM_OK = False

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("herg_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Cannot load model files: {e}")
        st.stop()

# -------------------------------
# EXTRACT FEATURES
# -------------------------------
def get_features(smiles):
    if not PUBCHEM_OK:
        st.error("PubChemPy not available")
        return None
    
    try:
        compounds = get_compounds(smiles, "smiles")
        if not compounds:
            return None
        c = compounds[0]
        
        return {
            "MolWt": c.molecular_weight or 0,
            "LogP": c.xlogp or 0,
            "TPSA": c.tpsa or 0,
            "HBD": c.hbond_donor_count or 0,
            "HBA": c.hbond_acceptor_count or 0,
            "RotatableBonds": c.rotatable_bond_count or 0,
            "HeavyAtoms": c.heavy_atom_count or 0,
            "Complexity": c.complexity or 0,
            "Charge": c.charge or 0,
            "ExactMass": c.exact_mass or 0,
        }
    except:
        return None

# -------------------------------
# MAIN APP
# -------------------------------
model, scaler = load_model()

st.title("🧬 hERG Cardiotoxicity Predictor")
st.markdown("Predict if a compound blocks the **hERG potassium channel** (causes cardiotoxicity)")

# Sidebar
st.sidebar.header("ℹ️ About")
st.sidebar.info("""
**Model:** Random Forest  
**Features:** PubChem descriptors  
**Input:** SMILES notation
""")

st.sidebar.header("📝 Examples")
st.sidebar.code("CC(=O)OC1=CC=CC=C1C(=O)O")
st.sidebar.caption("Aspirin")

# Input
smiles = st.text_input("Enter SMILES:", placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O")

if st.button("🔍 Predict", type="primary"):
    if not smiles.strip():
        st.warning("Please enter a SMILES string")
    else:
        with st.spinner("Analyzing..."):
            features = get_features(smiles.strip())
        
        if not features:
            st.error("❌ Invalid SMILES or PubChem lookup failed")
        else:
            # Predict
            X = pd.DataFrame([features])
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0]
            
            # Results
            st.header("Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", "TOXIC ⚠️" if pred == 1 else "SAFE ✅")
            c2.metric("Non-toxic %", f"{prob[0]*100:.1f}%")
            c3.metric("Toxic %", f"{prob[1]*100:.1f}%")
            
            # Features
            st.subheader("Molecular Features")
            df = pd.DataFrame(features.items(), columns=["Feature", "Value"])
            st.dataframe(df, hide_index=True)
            
            # Chart
            fig = px.bar(x=list(features.keys()), y=list(features.values()),
                        labels={"x": "Feature", "y": "Value"})
            st.plotly_chart(fig, use_container_width=True)
            
            # Warning
            if pred == 1:
                st.warning("⚠️ This compound may cause cardiac issues")
            else:
                st.success("✅ Low cardiotoxicity risk predicted")

st.markdown("---")
st.caption("Data: TDC | Features: PubChemPy")
