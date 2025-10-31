import streamlit as st

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="hERG Toxicity Predictor",
    page_icon="🧬",
    layout="wide"
)

import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    st.error("❌ RDKit is not available. Please check your installation.")
    st.stop()

# --------------------------------------------------
# LOAD MODEL AND SCALER
# --------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load('herg_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info("Please ensure `herg_model.pkl` and `scaler.pkl` are in the repository root.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------
def smiles_to_features(smiles):
    """Extract molecular features using RDKit"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
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
            'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
            'NumHeteroAtoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
            'MolMR': Descriptors.MolMR(mol),
            'HeavyAtomMolWt': Descriptors.HeavyAtomMolWt(mol),
            'ExactMolWt': Descriptors.ExactMolWt(mol),
            'NumValenceElectrons': Descriptors.NumValenceElectrons(mol)
        }
        return features, mol
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None, None

# --------------------------------------------------
# LIPINSKI'S RULE OF FIVE
# --------------------------------------------------
def lipinski_check(features):
    """Check Lipinski's Rule of Five violations"""
    violations = 0
    rules_violated = []
    
    if features['MolWt'] > 500:
        violations += 1
        rules_violated.append("Molecular Weight > 500")
    if features['LogP'] > 5:
        violations += 1
        rules_violated.append("LogP > 5")
    if features['NumHDonors'] > 5:
        violations += 1
        rules_violated.append("H-Bond Donors > 5")
    if features['NumHAcceptors'] > 10:
        violations += 1
        rules_violated.append("H-Bond Acceptors > 10")
    
    return violations, rules_violated

# --------------------------------------------------
# APP HEADER
# --------------------------------------------------
st.title("🧬 hERG Cardiotoxicity Predictor")
st.markdown("""
This tool predicts whether a compound blocks the **hERG potassium channel**, which is associated with cardiac toxicity.  
Model trained on **Therapeutics Data Commons (TDC)** hERG dataset using RDKit molecular descriptors.
""")

# Sidebar
st.sidebar.header("ℹ️ About")
st.sidebar.info("""
**Model:** Random Forest Classifier  
**Dataset:** TDC hERG Toxicity  
**Features:** RDKit Descriptors  
**Input:** SMILES notation  
**Output:** Toxic (1) or Non-toxic (0)
""")

st.sidebar.header("📝 Example SMILES")
st.sidebar.code("CCO", language="text")
st.sidebar.caption("Ethanol")
st.sidebar.code("CC(=O)OC1=CC=CC=C1C(=O)O", language="text")
st.sidebar.caption("Aspirin")
st.sidebar.code("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", language="text")
st.sidebar.caption("Caffeine")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model, scaler = load_model()
st.success("✅ Model loaded successfully!")

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
            placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O",
            key="smiles_predict"
        )

        if st.button("🔍 Predict Toxicity", type="primary"):
            if smiles_input.strip():
                with st.spinner("Analyzing molecule..."):
                    result = smiles_to_features(smiles_input.strip())
                
                if result is None or result[0] is None:
                    st.error("❌ Invalid SMILES notation or feature extraction failed.")
                else:
                    features, mol = result
                    
                    # Prepare features for model
                    features_df = pd.DataFrame([features])
                    
                    # Align columns with model scaler
                    try:
                        model_features = scaler.feature_names_in_
                        features_df = features_df.reindex(columns=model_features, fill_value=0)
                    except AttributeError:
                        # If scaler doesn't have feature_names_in_, use as is
                        pass
                    
                    features_scaled = scaler.transform(features_df)

                    # Make prediction
                    prediction = model.predict(features_scaled)[0]
                    prediction_proba = model.predict_proba(features_scaled)[0]

                    # Display results
                    st.subheader("🧠 Prediction Results")
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        if prediction == 1:
                            st.metric("Prediction", "TOXIC ⚠️", delta="High Risk", delta_color="inverse")
                        else:
                            st.metric("Prediction", "NON-TOXIC ✅", delta="Low Risk", delta_color="normal")
                    
                    with col_b:
                        st.metric("Non-toxic Confidence", f"{prediction_proba[0]:.2%}")
                    
                    with col_c:
                        st.metric("Toxic Confidence", f"{prediction_proba[1]:.2%}")

                    # Gauge chart for toxicity probability
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction_proba[1]*100,
                        title={'text': "Toxicity Probability (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "crimson" if prediction == 1 else "green"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                    # Molecular structure
                    st.subheader("🔬 Molecular Structure")
                    try:
                        img = Draw.MolToImage(mol, size=(400, 400))
                        st.image(img, caption="2D Molecular Structure", use_container_width=False)
                    except Exception as e:
                        st.warning(f"Could not generate molecule image: {e}")

                    # Lipinski rule check
                    st.subheader("💊 Drug-likeness (Lipinski's Rule of Five)")
                    violations, rules_violated = lipinski_check(features)
                    
                    if violations == 0:
                        st.success("✅ Follows Lipinski's Rule of Five - Good drug-like properties!")
                    else:
                        st.warning(f"⚠️ {violations} Lipinski violation(s) detected:")
                        for rule in rules_violated:
                            st.write(f"  • {rule}")
                        st.caption("Note: Some violations may be acceptable for certain drug classes")

                    # Molecular properties table
                    st.subheader("📋 Molecular Properties")
                    props_df = pd.DataFrame(features.items(), columns=['Property', 'Value'])
                    props_df['Value'] = props_df['Value'].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
                    st.dataframe(props_df, use_container_width=True, hide_index=True)

                    # Toxicity interpretation
                    st.subheader("🔍 Interpretation")
                    if prediction == 1:
                        st.error("""
                        ⚠️ **Warning: High hERG Toxicity Risk**
                        
                        This compound is predicted to block the hERG potassium channel, which may cause:
                        - **QT interval prolongation**
                        - **Cardiac arrhythmias** (Torsades de Pointes)
                        - **Potential cardiac arrest**
                        
                        **Recommended Actions:**
                        - Structural modification to reduce hERG binding
                        - Experimental hERG assay validation
                        - Consider alternative scaffolds
                        - Risk-benefit analysis if proceeding
                        """)
                    else:
                        st.success("""
                        ✅ **Low hERG Toxicity Risk**
                        
                        This compound is predicted to have minimal hERG channel blocking activity.
                        
                        **Next Steps:**
                        - Validate with in vitro hERG assays
                        - Continue with other toxicity screens
                        - Evaluate in broader safety panel
                        - Remember: This is one aspect of drug safety
                        """)
            else:
                st.warning("⚠️ Please enter a valid SMILES string")

    with col2:
        st.info("""
        **📖 How to use:**
        
        1. Enter a SMILES string in the input box
        2. Click "Predict Toxicity"
        3. Review the prediction and confidence scores
        4. Check molecular properties and drug-likeness
        5. Read the interpretation and recommendations
        
        **💡 Tips:**
        - Use standard SMILES notation
        - Check example SMILES in sidebar
        - Validate predictions experimentally
        """)

# --------------------------------------------------
# TAB 2: PROPERTIES
# --------------------------------------------------
with tabs[1]:
    st.header("📊 Molecular Property Visualization")
    
    smiles_viz = st.text_input(
        "Enter SMILES for property analysis:",
        value="CC(=O)OC1=CC=CC=C1C(=O)O",
        key="smiles_viz"
    )
    
    if smiles_viz.strip():
        result = smiles_to_features(smiles_viz.strip())
        
        if result and result[0]:
            features_viz, mol_viz = result
            
            # Properties dataframe
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Property Values")
                df_props = pd.DataFrame(features_viz.items(), columns=['Property', 'Value'])
                df_props['Value'] = df_props['Value'].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
                st.dataframe(df_props, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("Molecular Image")
                try:
                    img = Draw.MolToImage(mol_viz, size=(300, 300))
                    st.image(img, use_container_width=True)
                except:
                    st.warning("Could not generate image")
            
            # Bar chart of properties
            st.subheader("Property Distribution")
            fig_bar = px.bar(
                x=list(features_viz.keys()),
                y=list(features_viz.values()),
                labels={'x': 'Property', 'y': 'Value'},
                title="Molecular Descriptors",
                color=list(features_viz.values()),
                color_continuous_scale="viridis"
            )
            fig_bar.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Radar chart
            st.subheader("Molecular Descriptor Radar Plot")
            # Normalize values for radar chart
            values = list(features_viz.values())
            max_val = max(values) if max(values) > 0 else 1
            normalized = [v/max_val * 100 for v in values]
            
            fig_radar = go.Figure(go.Scatterpolar(
                r=normalized,
                theta=list(features_viz.keys()),
                fill='toself',
                name='Properties'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.error("❌ Invalid SMILES for visualization")

# --------------------------------------------------
# TAB 3: MODEL INFO
# --------------------------------------------------
with tabs[2]:
    st.header("ℹ️ Model Information")
    
    st.markdown("""
    ### 🔬 Technical Details
    
    **Algorithm:** Random Forest Classifier  
    **Feature Extraction:** RDKit molecular descriptors  
    **Dataset:** Therapeutics Data Commons (TDC) - hERG Toxicity Dataset  
    **Frameworks:** 
    - RDKit for molecular feature extraction
    - Scikit-learn for machine learning
    - Streamlit for web interface
    - Plotly for interactive visualizations
    
    ### 🎯 Model Purpose
    
    The hERG (human Ether-à-go-go-Related Gene) potassium channel is crucial for cardiac repolarization. 
    Blocking this channel can cause:
    - **Long QT Syndrome (LQTS)**
    - **Torsades de Pointes** (dangerous arrhythmia)
    - **Sudden cardiac death**
    
    This is one of the most common reasons for drug withdrawal from the market.
    
    ### 📊 Feature Set
    
    The model uses 16 molecular descriptors including:
    - **Physicochemical:** Molecular weight, LogP, TPSA
    - **Structural:** Ring counts, rotatable bonds, heavy atoms
    - **Electronic:** Valence electrons, molecular refractivity
    - **Topological:** Fraction CSP3, heteroatoms
    
    ### ⚠️ Disclaimer
    
    This tool is for **research and educational purposes only**. 
    - Predictions should be validated experimentally
    - Not a substitute for in vitro/in vivo testing
    - Part of a comprehensive drug safety evaluation
    - Always consult with medicinal chemistry experts
    
    ### 📚 References
    
    - [Therapeutics Data Commons](https://tdcommons.ai/)
    - [RDKit Documentation](https://www.rdkit.org/docs/)
    - [hERG Channel Information (FDA)](https://www.fda.gov/drugs)
    """)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🧬 Data Source: <a href='https://tdcommons.ai/' target='_blank'>Therapeutics Data Commons (TDC)</a></p>
    <p>⚗️ Powered by RDKit | 🤖 Machine Learning with Scikit-learn</p>
    <p>© 2025 hERG Cardiotoxicity Prediction Tool</p>
</div>
""", unsafe_allow_html=True)
