import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("="*60)
print("LOADING REAL TDC hERG DATASET")
print("="*60)

# Import TDC properly
from tdc.single_pred import Tox

# Load real hERG dataset
print("\n[1/5] Downloading hERG dataset from TDC...")
data = Tox(name='hERG')
df = data.get_data()

print(f"✓ Dataset loaded successfully!")
print(f"  - Total samples: {len(df)}")
print(f"  - Columns: {list(df.columns)}")
print(f"  - First few rows:\n{df.head()}")

# Import RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

def smiles_to_features(smiles):
    """Convert SMILES to molecular descriptors"""
    try:
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
            'NumHeavyAtoms': rdMolDescriptors.CalcNumHeavyAtoms(mol),
            'NumRings': rdMolDescriptors.CalcNumRings(mol),
        }
        return features
    except Exception as e:
        return None

print("\n[2/5] Extracting molecular features from SMILES...")
features_list = []
valid_indices = []
invalid_count = 0

total = len(df)
for idx, smiles in enumerate(df['Drug']):
    features = smiles_to_features(smiles)
    if features is not None:
        features_list.append(features)
        valid_indices.append(idx)
    else:
        invalid_count += 1
    
    if (idx + 1) % 500 == 0:
        print(f"  Processed {idx + 1}/{total} molecules... ({len(features_list)} valid)")

print(f"\n✓ Feature extraction complete!")
print(f"  - Valid molecules: {len(features_list)}")
print(f"  - Invalid/skipped: {invalid_count}")

# Create feature dataframe
features_df = pd.DataFrame(features_list)
y = df.loc[valid_indices, 'Y'].values

print(f"\n[3/5] Preparing dataset...")
print(f"  - Feature matrix shape: {features_df.shape}")
print(f"  - Features: {list(features_df.columns)}")
print(f"  - Class distribution:")
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    print(f"    Class {label}: {count} samples ({count/len(y)*100:.1f}%)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features_df, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  - Training set: {len(X_train)} samples")
print(f"  - Test set: {len(X_test)} samples")

# Scale features
print("\n[4/5] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled")

# Train model
print("\n[5/5] Training Random Forest model...")
print("  - Algorithm: Random Forest")
print("  - Trees: 100")
print("  - Max depth: 15")
print("  - Training in progress...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
model.fit(X_train_scaled, y_train)

print("✓ Training complete!")

# Evaluate
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print(f"\n{'='*60}")
print("CLASSIFICATION REPORT (Test Set)")
print('='*60)
print(classification_report(y_test, y_test_pred, 
                          target_names=['Non-toxic', 'Toxic'],
                          digits=4))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features_df.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*60)
print("TOP 5 MOST IMPORTANT FEATURES")
print("="*60)
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']:<20} {row['importance']:.4f}")

# Save everything
print("\n" + "="*60)
print("SAVING MODEL FILES")
print("="*60)

joblib.dump(model, 'herg_model.pkl')
print("✓ Saved: herg_model.pkl")

joblib.dump(scaler, 'scaler.pkl')
print("✓ Saved: scaler.pkl")

joblib.dump(list(features_df.columns), 'feature_names.pkl')
print("✓ Saved: feature_names.pkl")

# Save training info
training_info = {
    'dataset': 'TDC hERG',
    'total_samples': len(df),
    'valid_samples': len(features_list),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'train_accuracy': float(train_accuracy),
    'test_accuracy': float(test_accuracy),
    'features': list(features_df.columns),
    'model_type': 'RandomForestClassifier',
    'n_estimators': 100
}
joblib.dump(training_info, 'training_info.pkl')
print("✓ Saved: training_info.pkl")

print("\n" + "="*60)
print("✓ MODEL TRAINING COMPLETE!")
print("="*60)
print(f"\nFiles created in current directory:")
print("  - herg_model.pkl (trained model)")
print("  - scaler.pkl (feature scaler)")
print("  - feature_names.pkl (feature list)")
print("  - training_info.pkl (training metadata)")
print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
print("\nYou can now run: streamlit run app.py")
print("="*60)