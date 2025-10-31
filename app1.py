# app1.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO

# ─────────────────────────────────────────────────────────────
# ✅ Safe RDKit Import (Cloud compatible)
# ─────────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Draw
    rdkit_available = True
except ImportError:
    rdkit_available = False
    st.warning("⚠️ RDKit is not available in this environment. Molecular features and visualization will be limited.")

# ─────────────────────────────────────────────────────────────
# ✅ Streamlit Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🧬 hERG Cardiotoxicity Predictor",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# ✅ Load Model and Scaler
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("herg_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


# ─────────────────────────────────────────────────────────────
# ✅ Feature Extraction from SMILES
# ─────────────────────────────────────────────────────────────
def smiles_to_features(smiles):
    if not rdkit_available:
        return None
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
        'NumRings': rdMolDescriptors.CalcNumRings(mol)
    }
    return features


# ─────────────────────────────────────────────────────────────
# ✅ UI Header
# ─────────────────────────────────────────────────────────────
st.title("🧬 hERG Cardiotoxicity Predictor")

st.markdown("""
This web application predicts whether a small-molecule compound is likely to block the **hERG potassium channel**, 
a major cause of drug-induced cardiac arrhythmias.  
Built using a Random Forest model trained on the **Therapeutics Data Commons (TDC)** hERG dataset.
""")

# ─────────────────────────────────────────────────────────────
# ✅ Sidebar
# ─────────────────────────────────────────────────────────────
st.sidebar.header("ℹ️ About the Model")
st.sidebar.info("""
**Algorithm:** Random Forest Classifier  
**Dataset:** TDC hERG Toxicity  
**Input:** SMILES molecular notation  
**Output:** Toxic (1) or Non-toxic (0)
""")

st.sidebar.header("🧪 Example SMILES")
st.sidebar.code("CCO (Ethanol)")
st.sidebar.code("CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)")
st.sidebar.code("CN1C=NC2=C1C(=O)N(C(=O)N2C)C (Caffeine)")

# ─────────────────────────────────────────────────────────────
# ✅ Main App
# ─────────────────────────────────────────────────────────────
try:
    model, scaler = load_model()
    st.success("✅ Model loaded successfully!")

    st.header("🔬 Enter Molecule Information")

    col1, col2 = st.columns([2, 1])

    with col1:
        smiles_input = st.text_input(
            "Enter SMILES notation:",
            placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O",
            help="SMILES (Simplified Molecular Input Line Entry System) encodes molecular structures."
        )

        if st.button("🔍 Predict Toxicity", type="primary"):
            if not smiles_input:
                st.warning("Please enter a SMILES string first.")
            else:
                with st.spinner("Analyzing molecule…"):
                    if not rdkit_available:
                        st.error("❌ RDKit not available — cannot analyze molecule.")
                    else:
                        mol = Chem.MolFromSmiles(smiles_input)
                        if mol is None:
                            st.error("❌ Invalid SMILES notation.")
                        else:
                            features = smiles_to_features(smiles_input)
                            if features is None:
                                st.error("❌ Could not extract molecular features.")
                            else:
                                features_df = pd.DataFrame([features])
                                features_scaled = scaler.transform(features_df)
                                prediction = model.predict(features_scaled)[0]
                                prediction_proba = model.predict_proba(features_scaled)[0]

                                st.header("🧠 Prediction Results")
                                col_a, col_b, col_c = st.columns(3)

                                with col_a:
                                    st.metric("Prediction",
                                              "TOXIC ⚠️" if prediction == 1 else "NON-TOXIC ✅")
                                with col_b:
                                    st.metric("Confidence (Non-toxic)",
                                              f"{prediction_proba[0]:.2%}")
                                with col_c:
                                    st.metric("Confidence (Toxic)",
                                              f"{prediction_proba[1]:.2%}")

                                # 2D Structure
                                st.subheader("💊 Molecular Structure")
                                img = Draw.MolToImage(mol, size=(400, 400))
                                st.image(img, caption="2D Molecular Structure")

                                # Descriptor Table
                                st.subheader("📊 Molecular Properties")
                                props_df = pd.DataFrame([features]).T
                                props_df.columns = ["Value"]
                                st.dataframe(props_df, use_container_width=True)

                                # Radar Chart
                                fig = px.line_polar(
                                    r=list(features.values()),
                                    theta=list(features.keys()),
                                    line_close=True,
                                    title="Descriptor Profile Radar Plot"
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Final message
                                if prediction == 1:
                                    st.warning("""
                                    ⚠️ **Warning:** This compound is predicted to block the hERG channel.  
                                    It may cause cardiac arrhythmias or QT interval prolongation.
                                    """)
                                else:
                                    st.success("""
                                    ✅ **Good News:** This molecule is predicted to be safe with respect to hERG toxicity.  
                                    *(Note: This is only one aspect of drug safety.)*
                                    """)

    with col2:
        st.info("""
        **How to use:**  
        1️⃣ Enter a valid SMILES notation  
        2️⃣ Click ‘Predict Toxicity’  
        3️⃣ Review results & molecular profile  

        **Note:** Model predictions are for research use only and not for clinical decisions.
        """)

except FileNotFoundError:
    st.error("""
    ❌ Model files not found!  
    Please train and save the model first using:
    ```bash
    python train_model.py
    ```
    """)

# ─────────────────────────────────────────────────────────────
# ✅ Footer
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center'>
  <p>Data source: <a href='https://tdcommons.ai/' target='_blank'>Therapeutics Data Commons (TDC)</a></p>
  <p>Model trained on hERG toxicity dataset</p>
</div>
""", unsafe_allow_html=True)
