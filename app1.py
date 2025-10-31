import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Draw
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="hERG Toxicity Predictor",
    page_icon="🧬",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL AND SCALER
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load('herg_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    features = {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': rdMolDescriptors.CalcNumHBD(mol),
        'NumHAcceptors': rdMolDescriptors.CalcNumHBA(mol),
        'TPSA': rdMolDescriptors.CalcTPSA(mol),
        'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(mol),
        'NumHeavyAtoms': mol.GetNumHeavyAtoms(),
        'NumRings': rdMolDescriptors.CalcNumRings(mol),
        # Extra descriptors for robustness
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
        'NumHeteroAtoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
        'MolMR': Descriptors.MolMR(mol),
        'HeavyAtomMolWt': Descriptors.HeavyAtomMolWt(mol),
        'ExactMolWt': Descriptors.ExactMolWt(mol),
        'NumValenceElectrons': Descriptors.NumValenceElectrons(mol)
    }
    return features

# --------------------------------------------------
# LIPINSKI'S RULE OF FIVE
# --------------------------------------------------
def lipinski_check(features):
    violations = 0
    if features['MolWt'] > 500: violations += 1
    if features['LogP'] > 5: violations += 1
    if features['NumHDonors'] > 5: violations += 1
    if features['NumHAcceptors'] > 10: violations += 1
    return violations

# --------------------------------------------------
# APP HEADER
# --------------------------------------------------
st.title("🧬 hERG Cardiotoxicity Predictor")
st.markdown("""
This tool predicts whether a compound blocks the **hERG potassium channel**, which is associated with cardiac toxicity.  
Model trained on **Therapeutics Data Commons (TDC)** hERG dataset.
""")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
**Model:** Random Forest Classifier  
**Dataset:** TDC hERG Toxicity  
**Input:** SMILES molecular notation  
**Output:** Toxic (1) or Non-toxic (0)
""")

st.sidebar.header("Example SMILES")
st.sidebar.code("CCO (Ethanol)")
st.sidebar.code("CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)")
st.sidebar.code("CN1C=NC2=C1C(=O)N(C(=O)N2C)C (Caffeine)")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
try:
    model, scaler = load_model()
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("""
    ❌ Model files not found!  
    Please ensure `herg_model.pkl` and `scaler.pkl` are in the same directory.
    """)
    st.stop()

# --------------------------------------------------
# CREATE TABS
# --------------------------------------------------
tabs = st.tabs(["🔍 Predict", "📊 Properties", "ℹ️ Model Info"])

# --------------------------------------------------
# TAB 1: PREDICTION
# --------------------------------------------------
with tabs[0]:
    st.header("Enter Molecule Information")

    col1, col2 = st.columns([2, 1])

    with col1:
        smiles_input = st.text_input(
            "Enter SMILES notation:",
            placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O"
        )

        if st.button("🔍 Predict Toxicity", type="primary"):
            if smiles_input:
                mol = Chem.MolFromSmiles(smiles_input)
                if mol is None:
                    st.error("❌ Invalid SMILES notation.")
                else:
                    features = smiles_to_features(smiles_input)
                    if features is None:
                        st.error("❌ Could not extract molecular features.")
                    else:
                        features_df = pd.DataFrame([features])
                        # Align columns with model scaler
                        model_features = scaler.feature_names_in_
                        features_df = features_df.reindex(columns=model_features, fill_value=0)
                        features_scaled = scaler.transform(features_df)

                        prediction = model.predict(features_scaled)[0]
                        prediction_proba = model.predict_proba(features_scaled)[0]

                        st.subheader("Prediction Results")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Prediction", "TOXIC ⚠️" if prediction == 1 else "NON-TOXIC ✅")
                        with col_b:
                            st.metric("Confidence (Non-toxic)", f"{prediction_proba[0]:.2%}")
                        with col_c:
                            st.metric("Confidence (Toxic)", f"{prediction_proba[1]:.2%}")

                        # Gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prediction_proba[1]*100,
                            title={'text': "Toxicity Probability (%)"},
                            gauge={'axis': {'range': [0,100]}, 'bar': {'color': "crimson"}}
                        ))
                        st.plotly_chart(fig, use_container_width=True)

                        # Molecule structure
                        st.subheader("Molecular Structure")
                        img = Draw.MolToImage(mol, size=(350, 350))
                        st.image(img, caption="2D Molecular Structure")

                        # Lipinski rule check
                        violations = lipinski_check(features)
                        if violations == 0:
                            st.success("✅ Follows Lipinski's Rule of Five (Drug-like)")
                        else:
                            st.warning(f"⚠️ {violations} Lipinski violations detected (possible low bioavailability)")

                        # Toxicity warning
                        if prediction == 1:
                            st.error("""
                            ⚠️ **Warning:** Predicted to block hERG channel — potential cardiac toxicity risk.
                            """)
                        else:
                            st.success("""
                            ✅ **Good News:** This molecule is predicted to be non-toxic for hERG blocking.
                            """)
            else:
                st.warning("Please enter a SMILES string.")

    with col2:
        st.info("""
        **How to use:**  
        1. Enter a SMILES string  
        2. Click “Predict Toxicity”  
        3. Review toxicity and properties  
        """)

# --------------------------------------------------
# TAB 2: PROPERTIES
# --------------------------------------------------
with tabs[1]:
    st.header("Molecular Property Visualization")
    smiles_example = st.text_input("Enter SMILES for property chart:", "CCO")
    mol2 = Chem.MolFromSmiles(smiles_example)
    if mol2:
        features2 = smiles_to_features(smiles_example)
        df_props = pd.DataFrame([features2]).T
        df_props.columns = ['Value']
        st.dataframe(df_props, use_container_width=True)

        # Radar chart
        fig = px.line_polar(
            r=list(features2.values()),
            theta=list(features2.keys()),
            line_close=True,
            title="Molecular Descriptor Radar Plot"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Invalid SMILES for property chart.")

# --------------------------------------------------
# TAB 3: MODEL INFO
# --------------------------------------------------
with tabs[2]:
    st.header("Model Information")
    st.markdown("""
    **Algorithm:** Random Forest Classifier  
    **Dataset:** Therapeutics Data Commons (TDC) – hERG Toxicity  
    **Frameworks:** RDKit, Scikit-learn, Streamlit  

    The model predicts if a molecule can block the **hERG potassium channel**, a known cause of **cardiac arrhythmias**.
    """)
    st.markdown("---")
    st.markdown("""
    **Developer Notes:**  
    - Extendable with batch prediction and 3D visualization.  
    - For scientific validation, combine predictions with experimental assays.
    """)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Data Source: <a href='https://tdcommons.ai/'>Therapeutics Data Commons (TDC)</a></p>
    <p>Model trained on real hERG toxicity data</p>
</div>
""", unsafe_allow_html=True)
