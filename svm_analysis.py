import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('public/blood_cancer_diseases_dataset.csv')

print("=== Original Dataset Overview ===")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Data Preprocessing
print("\n=== DATA PREPROCESSING ===")

# 1. Handle Missing Values
print("Missing values before preprocessing:")
print(df.isnull().sum())

# Handle missing values with imputation instead of dropping all rows
# For numeric columns, use median imputation
numeric_columns = ['Age', 'Total WBC count(/cumm)', 'Platelet Count( (/cumm)']
for col in numeric_columns:
    if col in df.columns:
        # Convert to numeric, replacing any non-numeric values with NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill NaN values with median
        df[col].fillna(df[col].median(), inplace=True)

# For categorical columns, use mode imputation
categorical_columns = ['Genetic_Data(BCR-ABL, FLT3)', 'Side_Effects']
for col in categorical_columns:
    if col in df.columns:
        # Fill NaN values with mode (most frequent value)
        mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        df[col].fillna(mode_value, inplace=True)

# Convert gender to binary encoding
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Convert treatment types to numerical encoding
treatment_types = df['Treatment_Type(Chemotherapy, Radiation)'].unique()
treatment_mapping = {treatment: idx for idx, treatment in enumerate(treatment_types)}
df['Treatment_Type(Chemotherapy, Radiation)'] = df['Treatment_Type(Chemotherapy, Radiation)'].map(treatment_mapping)

# Convert bone marrow aspiration to numerical encoding
df['Bone Marrow Aspiration(Positive / Negative / Not Done)'] = df['Bone Marrow Aspiration(Positive / Negative / Not Done)'].map({
    'Positive': 1,
    'Negative': 0,
    'Not Done': -1
})

# Convert SPEP to numerical encoding
df['Serum Protein Electrophoresis (SPEP)(Normal / Abnormal)'] = df['Serum Protein Electrophoresis (SPEP)(Normal / Abnormal)'].map({
    'Normal': 0,
    'Abnormal': 1,
    'Not Done': -1
})

# Convert lymph node biopsy to numerical encoding
df['Lymph Node Biopsy(Positive / Negative / Not Done)'] = df['Lymph Node Biopsy(Positive / Negative / Not Done)'].map({
    'Positive': 1,
    'Negative': 0,
    'Not Done': -1
})

# Convert lumbar puncture to numerical encoding
lumbar_puncture_values = df['Lumbar Puncture (Spinal Tap)'].unique()
lumbar_puncture_mapping = {val: idx for idx, val in enumerate(lumbar_puncture_values)}
df['Lumbar Puncture (Spinal Tap)'] = df['Lumbar Puncture (Spinal Tap)'].map(lumbar_puncture_mapping)

# Convert treatment outcome to numerical encoding
treatment_outcome_values = df['Treatment_Outcome'].unique()
treatment_outcome_mapping = {val: idx for idx, val in enumerate(treatment_outcome_values)}
df['Treatment_Outcome'] = df['Treatment_Outcome'].map(treatment_outcome_mapping)

# Convert genetic data to numerical encoding
genetic_data_values = df['Genetic_Data(BCR-ABL, FLT3)'].unique()
genetic_data_mapping = {val: idx for idx, val in enumerate(genetic_data_values)}
df['Genetic_Data(BCR-ABL, FLT3)'] = df['Genetic_Data(BCR-ABL, FLT3)'].map(genetic_data_mapping)

# Convert side effects to numerical encoding
side_effects_values = df['Side_Effects'].unique()
side_effects_mapping = {val: idx for idx, val in enumerate(side_effects_values)}
df['Side_Effects'] = df['Side_Effects'].map(side_effects_mapping)

# Convert diagnosis result to numerical encoding
diagnosis_result_values = df['Diagnosis_Result'].unique()
diagnosis_result_mapping = {val: idx for idx, val in enumerate(diagnosis_result_values)}
df['Diagnosis_Result'] = df['Diagnosis_Result'].map(diagnosis_result_mapping)

# Convert comments to numerical encoding
comments_values = df['Comments'].unique()
comments_mapping = {val: idx for idx, val in enumerate(comments_values)}
df['Comments'] = df['Comments'].map(comments_mapping)

# Convert cancer types to numerical encoding (this will be our target variable)
cancer_types = df['Cancer_Type(AML, ALL, CLL)'].unique()
cancer_mapping = {cancer: idx for idx, cancer in enumerate(cancer_types)}
df['Cancer_Type(AML, ALL, CLL)'] = df['Cancer_Type(AML, ALL, CLL)'].map(cancer_mapping)

print("\nMissing values after preprocessing:")
print(df.isnull().sum())

# Ensure all columns are numeric
print("\nData types after preprocessing:")
print(df.dtypes)

# Remove any remaining rows with NaN values
df.dropna(inplace=True)
print(f"\nDataset shape after removing remaining NaN values: {df.shape}")

# 2. Feature Selection
print("\n=== FEATURE SELECTION ===")
X = df.drop('Cancer_Type(AML, ALL, CLL)', axis=1)
y = df['Cancer_Type(AML, ALL, CLL)']

# Ensure X contains only numeric data
print("Feature matrix shape:", X.shape)
print("Feature matrix dtypes:")
print(X.dtypes)

# Apply SelectKBest for feature selection
selector = SelectKBest(score_func=f_classif, k=min(10, X.shape[1]))  # Use min to avoid selecting more features than available
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print(f"Selected features: {selected_features.tolist()}")

# 3. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_new)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# 5. Model Training with Grid Search
print("\n=== MODEL TRAINING ===")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

svm = SVC(random_state=42)
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Use the best model
best_model = grid_search.best_estimator_

# 6. Model Evaluation
print("\n=== MODEL EVALUATION ===")
y_pred = best_model.predict(X_test)

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
# Create reverse mapping for cancer types for better readability
reverse_cancer_mapping = {v: k for k, v in cancer_mapping.items()}

# Get unique classes present in y_test and y_pred
unique_classes = np.unique(np.concatenate([y_test, y_pred]))
target_names = [reverse_cancer_mapping[i] for i in sorted(unique_classes)]

print(f"Unique classes in test set: {unique_classes}")
print(f"Target names: {target_names}")
print(classification_report(y_test, y_pred, labels=unique_classes, target_names=target_names))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Cross-validation
cv_scores = cross_val_score(best_model, X_scaled, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 7. Feature Importance using Random Forest
print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': selected_features, 
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance, x='Importance', y='Feature')
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('public/svm_analysis/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot confusion matrix
plt.figure(figsize=(10, 8))
# Get unique classes for proper labeling
unique_classes = np.unique(np.concatenate([y_test, y_pred]))
target_names = [reverse_cancer_mapping[i] for i in sorted(unique_classes)]

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('public/svm_analysis/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Save Results
print("\n=== SAVING RESULTS ===")
# Save the best model
joblib.dump(best_model, 'public/svm_analysis/svm_model.pkl')
print("Model saved as 'public/svm_analysis/svm_model.pkl'.")

# Save the scaler
joblib.dump(scaler, 'public/svm_analysis/scaler.pkl')
print("Scaler saved as 'public/svm_analysis/scaler.pkl'.")

# Save the feature selector
joblib.dump(selector, 'public/svm_analysis/feature_selector.pkl')
print("Feature selector saved as 'public/svm_analysis/feature_selector.pkl'.")

# Save feature importance
feature_importance.to_csv('public/svm_analysis/feature_importance.csv', index=False)
print("Feature importance saved as 'public/svm_analysis/feature_importance.csv'.")

# Save the processed dataset
df.to_csv('public/svm_analysis/processed_blood_cancer_dataset.csv', index=False)
print("Processed dataset saved as 'public/svm_analysis/processed_blood_cancer_dataset.csv'.")

# Save all mappings
mappings = {
    'treatment_mapping': treatment_mapping,
    'cancer_mapping': cancer_mapping,
    'side_effects_mapping': side_effects_mapping,
    'comments_mapping': comments_mapping,
    'lumbar_puncture_mapping': lumbar_puncture_mapping,
    'treatment_outcome_mapping': treatment_outcome_mapping,
    'genetic_data_mapping': genetic_data_mapping,
    'diagnosis_result_mapping': diagnosis_result_mapping,
    'selected_features': selected_features.tolist()
}
joblib.dump(mappings, 'public/svm_analysis/mappings.pkl')
print("Mappings saved as 'public/svm_analysis/mappings.pkl'.")

# Create a summary report
summary = {
    'Dataset_Shape': df.shape,
    'Selected_Features': selected_features.tolist(),
    'Model_Performance': {
        'Accuracy': accuracy,
        'Best_Parameters': grid_search.best_params_,
        'CV_Mean_Score': cv_scores.mean(),
        'CV_Std_Score': cv_scores.std()
    },
    'Cancer_Types': list(cancer_mapping.keys()),
    'Feature_Importance': feature_importance.to_dict('records')
}

import json
with open('public/svm_analysis/analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Analysis summary saved as 'public/svm_analysis/analysis_summary.json'.")

print("\n=== ANALYSIS COMPLETE ===")
print(f"Final model accuracy: {accuracy:.4f}")
print(f"Number of features used: {len(selected_features)}")
print(f"Cancer types detected: {list(cancer_mapping.keys())}")