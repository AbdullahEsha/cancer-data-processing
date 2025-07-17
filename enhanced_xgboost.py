import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
from sklearn.utils import resample
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('public/blood_cancer_diseases_dataset.csv')

print("=== ENHANCED BLOOD CANCER CLASSIFICATION MODEL WITH XGBOOST ===")
print(f"Original dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# STEP 1: ADVANCED DATA PREPROCESSING
print("\n=== ADVANCED DATA PREPROCESSING ===")

# Create a copy for preprocessing
df_processed = df.copy()

# 1. Handle target variable first - filter out only the main cancer types
main_cancer_types = ['AML', 'ALL', 'CLL', 'Multiple Myeloma', 'CML']
df_processed = df_processed[df_processed['Cancer_Type(AML, ALL, CLL)'].isin(main_cancer_types)]
print(f"Filtered dataset shape (main cancer types only): {df_processed.shape}")

# 2. Intelligent feature selection - Remove potentially problematic columns
columns_to_drop = [
    'Comments',
    'Diagnosis_Result',
    'Treatment_Outcome',
    'Treatment_Type(Chemotherapy, Radiation)',
    'Side_Effects'
]
relevant_features = [
    'Age',
    'Gender',
    'Total WBC count(/cumm)',
    'Platelet Count( (/cumm)',
    'Bone Marrow Aspiration(Positive / Negative / Not Done)',
    'Serum Protein Electrophoresis (SPEP)(Normal / Abnormal)',
    'Lymph Node Biopsy(Positive / Negative / Not Done)',
    'Lumbar Puncture (Spinal Tap)',
    'Genetic_Data(BCR-ABL, FLT3)',
    'Cancer_Type(AML, ALL, CLL)'  # Target variable
]
df_processed = df_processed[relevant_features]

# 3. Advanced missing value handling and outlier capping
numeric_features = ['Age', 'Total WBC count(/cumm)', 'Platelet Count( (/cumm)']
for col in numeric_features:
    if col in df_processed.columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)
        df_processed[col].fillna(df_processed[col].median(), inplace=True)

# 4. Improved categorical encoding
df_processed['Gender'] = df_processed['Gender'].map({'Male': 0, 'Female': 1})
df_processed['Bone Marrow Aspiration(Positive / Negative / Not Done)'] = df_processed['Bone Marrow Aspiration(Positive / Negative / Not Done)'].map({
    'Positive': 2, 'Negative': 0, 'Not Done': 1
})
df_processed['Serum Protein Electrophoresis (SPEP)(Normal / Abnormal)'] = df_processed['Serum Protein Electrophoresis (SPEP)(Normal / Abnormal)'].map({
    'Normal': 0, 'Abnormal': 1, 'Not Done': 0.5
})
df_processed['Lymph Node Biopsy(Positive / Negative / Not Done)'] = df_processed['Lymph Node Biopsy(Positive / Negative / Not Done)'].map({
    'Positive': 2, 'Negative': 0, 'Not Done': 1
})
lumbar_puncture_mapping = {val: i for i, val in enumerate(df_processed['Lumbar Puncture (Spinal Tap)'].unique())}
df_processed['Lumbar Puncture (Spinal Tap)'] = df_processed['Lumbar Puncture (Spinal Tap)'].map(lumbar_puncture_mapping)
genetic_data_mapping = {val: i for i, val in enumerate(df_processed['Genetic_Data(BCR-ABL, FLT3)'].unique())}
df_processed['Genetic_Data(BCR-ABL, FLT3)'] = df_processed['Genetic_Data(BCR-ABL, FLT3)'].map(genetic_data_mapping)

# Fill any remaining missing values for numeric columns only
numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df_processed[col].fillna(df_processed[col].median(), inplace=True)
categorical_cols = df_processed.select_dtypes(include=['object']).columns
for col in categorical_cols:
    mode_value = df_processed[col].mode()
    if len(mode_value) > 0:
        df_processed[col].fillna(mode_value[0], inplace=True)
    else:
        df_processed[col].fillna('Unknown', inplace=True)

# 5. Feature Engineering
df_processed['WBC_Platelet_Ratio'] = df_processed['Total WBC count(/cumm)'] / (df_processed['Platelet Count( (/cumm)'] + 1)
df_processed['Age_Group'] = pd.cut(df_processed['Age'], bins=[0, 30, 60, 100], labels=[0, 1, 2])
df_processed['High_WBC'] = (df_processed['Total WBC count(/cumm)'] > df_processed['Total WBC count(/cumm)'].quantile(0.75)).astype(int)
df_processed['Low_Platelet'] = (df_processed['Platelet Count( (/cumm)'] < df_processed['Platelet Count( (/cumm)'].quantile(0.25)).astype(int)
df_processed['Age_Group'] = df_processed['Age_Group'].astype(int)

# 6. Prepare features and target
X = df_processed.drop('Cancer_Type(AML, ALL, CLL)', axis=1)
y = df_processed['Cancer_Type(AML, ALL, CLL)']

# Make all features numeric
for col in X.columns:
    if X[col].dtype == 'object':
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
X = X_imputed

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 7. Class balancing (simple oversampling)
def balance_classes(X, y):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    df_balanced = pd.concat([X, y.rename('target')], axis=1)
    unique_classes = sorted(df_balanced['target'].unique())
    class_dfs = []
    for cls in unique_classes:
        class_df = df_balanced[df_balanced['target'] == cls]
        class_dfs.append(class_df)
    max_size = max(len(df) for df in class_dfs)
    balanced_dfs = []
    for class_df in class_dfs:
        if len(class_df) < max_size:
            oversampled = resample(class_df, replace=True, n_samples=max_size, random_state=42)
            balanced_dfs.append(oversampled)
        else:
            balanced_dfs.append(class_df)
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    X_balanced = df_balanced.drop('target', axis=1)
    y_balanced = df_balanced['target']
    return X_balanced, y_balanced

X_balanced, y_balanced = balance_classes(X, y_encoded)
imputer_balanced = SimpleImputer(strategy='median')
X_balanced_clean = pd.DataFrame(imputer_balanced.fit_transform(X_balanced), columns=X_balanced.columns)
X_balanced_final = X_balanced_clean.values
y_balanced_final = y_balanced.values

# 8. Feature selection (mutual info)
selector = SelectKBest(score_func=mutual_info_classif, k=min(8, X_balanced_final.shape[1]))
X_selected = selector.fit_transform(X_balanced_final, y_balanced_final)
selected_features = X_balanced.columns[selector.get_support()]

# 9. Scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_selected)

# 10. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_balanced_final, test_size=0.2, random_state=42, stratify=y_balanced_final
)

# === XGBOOST MODEL TRAINING ===
print("\n=== XGBOOST MODEL TRAINING ===")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(xgb_model, param_grid, scoring='accuracy', cv=cv, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_xgb = grid_search.best_estimator_
print(f"Best XGBoost parameters: {grid_search.best_params_}")

# Evaluate XGBoost
y_pred_xgb = best_xgb.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

print("\nClassification Report (XGBoost):")
target_names = label_encoder.classes_
print(classification_report(y_test, y_pred_xgb, target_names=target_names))
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print("\nConfusion Matrix (XGBoost):")
print(cm_xgb)

# Cross-validation scores
cv_scores = cross_val_score(best_xgb, X_scaled, y_balanced_final, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance (from XGBoost)
importances = best_xgb.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance (XGBoost):")
print(feature_importance)

# Visualizations
import os
os.makedirs('public/enhanced_xgboost_analysis', exist_ok=True)
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance (XGBoost)', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig('public/enhanced_xgboost_analysis/xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - XGBoost', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('public/enhanced_xgboost_analysis/xgboost_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Save model & artifacts
joblib.dump(best_xgb, 'public/enhanced_xgboost_analysis/xgboost_model.pkl')
joblib.dump(scaler, 'public/enhanced_xgboost_analysis/scaler.pkl')
joblib.dump(selector, 'public/enhanced_xgboost_analysis/feature_selector.pkl')
joblib.dump(label_encoder, 'public/enhanced_xgboost_analysis/label_encoder.pkl')
feature_importance.to_csv('public/enhanced_xgboost_analysis/xgboost_feature_importance.csv', index=False)

# Save summary
enhanced_summary = {
    'Dataset_Shape': df_processed.shape,
    'Selected_Features': selected_features.tolist(),
    'Model_Performance': {
        'XGBoost_Accuracy': xgb_accuracy,
        'CV_Mean_Score': cv_scores.mean(),
        'CV_Std_Score': cv_scores.std()
    },
    'Cancer_Types': target_names.tolist(),
    'Feature_Importance': feature_importance.to_dict('records'),
    'Preprocessing_Steps': [
        'Removed irrelevant/leaky features',
        'Advanced outlier handling',
        'Intelligent categorical encoding',
        'Feature engineering',
        'Class balancing',
        'Mutual information feature selection',
        'Robust scaling'
    ]
}
import json
with open('public/enhanced_xgboost_analysis/xgboost_analysis_summary.json', 'w') as f:
    json.dump(enhanced_summary, f, indent=2)

print("\n=== XGBOOST ENHANCEMENT COMPLETE ===")
print(f"XGBoost accuracy: {xgb_accuracy:.2%}")
print(f"Selected features: {len(selected_features)}")
print(f"Cancer types: {target_names}")

# Prediction function for new data
def predict_cancer_type_xgb(age, gender, wbc_count, platelet_count, bone_marrow, spep, lymph_node, lumbar_puncture, genetic_data):
    """
    Function to predict cancer type for new patient data using XGBoost
    """
    # Prepare input data
    input_data = np.array([[age, gender, wbc_count, platelet_count, bone_marrow, spep, lymph_node, lumbar_puncture, genetic_data]])
    wbc_platelet_ratio = wbc_count / (platelet_count + 1)
    age_group = 0 if age <= 30 else 1 if age <= 60 else 2
    high_wbc = 1 if wbc_count > X['Total WBC count(/cumm)'].quantile(0.75) else 0
    low_platelet = 1 if platelet_count < X['Platelet Count( (/cumm)'].quantile(0.25) else 0
    input_data = np.append(input_data, [[wbc_platelet_ratio, age_group, high_wbc, low_platelet]], axis=1)
    input_selected = selector.transform(input_data)
    input_scaled = scaler.transform(input_selected)
    prediction = best_xgb.predict(input_scaled)[0]
    probability = best_xgb.predict_proba(input_scaled)[0]
    predicted_cancer = label_encoder.inverse_transform([prediction])[0]
    confidence = max(probability)
    return predicted_cancer, confidence

print("\nModel ready for predictions with XGBoost!")
print("Use predict_cancer_type_xgb() function for new predictions.")

# Training accuracy
# train_pred_xgb = best_xgb.predict(X_train)
# train_accuracy_xgb = accuracy_score(y_train, train_pred_xgb)
# print(f"Training Accuracy (XGBoost): {train_accuracy_xgb:.4f}")