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

print("=== ENHANCED BLOOD CANCER CLASSIFICATION MODEL WITH IMPROVED XGBOOST ===")
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

# 4. Improved categorical encoding with better mappings
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

# Handle lumbar puncture and genetic data with more meaningful encoding
lumbar_puncture_mapping = {val: i for i, val in enumerate(df_processed['Lumbar Puncture (Spinal Tap)'].unique())}
df_processed['Lumbar Puncture (Spinal Tap)'] = df_processed['Lumbar Puncture (Spinal Tap)'].map(lumbar_puncture_mapping)

genetic_data_mapping = {val: i for i, val in enumerate(df_processed['Genetic_Data(BCR-ABL, FLT3)'].unique())}
df_processed['Genetic_Data(BCR-ABL, FLT3)'] = df_processed['Genetic_Data(BCR-ABL, FLT3)'].map(genetic_data_mapping)

# Fill any remaining missing values
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

# 5. Enhanced Feature Engineering (removing Low_Platelet as it had 0 importance)
df_processed['WBC_Platelet_Ratio'] = df_processed['Total WBC count(/cumm)'] / (df_processed['Platelet Count( (/cumm)'] + 1)
df_processed['Age_Group'] = pd.cut(df_processed['Age'], bins=[0, 30, 60, 100], labels=[0, 1, 2])
df_processed['High_WBC'] = (df_processed['Total WBC count(/cumm)'] > df_processed['Total WBC count(/cumm)'].quantile(0.75)).astype(int)

# NEW FEATURES for better discrimination
df_processed['WBC_Age_Interaction'] = df_processed['Total WBC count(/cumm)'] * df_processed['Age']
df_processed['Platelet_Age_Interaction'] = df_processed['Platelet Count( (/cumm)'] * df_processed['Age']
df_processed['Combined_Test_Score'] = (
    df_processed['Bone Marrow Aspiration(Positive / Negative / Not Done)'] + 
    df_processed['Lymph Node Biopsy(Positive / Negative / Not Done)'] + 
    df_processed['Serum Protein Electrophoresis (SPEP)(Normal / Abnormal)']
)

# Log transformation for skewed features
df_processed['Log_WBC'] = np.log1p(df_processed['Total WBC count(/cumm)'])
df_processed['Log_Platelet'] = np.log1p(df_processed['Platelet Count( (/cumm)'])

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

# 7. Improved class balancing using SMOTE-like approach
def enhanced_balance_classes(X, y):
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
    
    # Use median size instead of max to avoid over-balancing
    sizes = [len(df) for df in class_dfs]
    target_size = int(np.median(sizes) * 1.2)  # 20% above median
    
    balanced_dfs = []
    for class_df in class_dfs:
        if len(class_df) < target_size:
            # Oversample with some noise
            oversampled = resample(class_df, replace=True, n_samples=target_size, random_state=42)
            balanced_dfs.append(oversampled)
        else:
            # Undersample if significantly larger
            if len(class_df) > target_size * 1.5:
                undersampled = resample(class_df, replace=False, n_samples=target_size, random_state=42)
                balanced_dfs.append(undersampled)
            else:
                balanced_dfs.append(class_df)
    
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X_balanced = df_balanced.drop('target', axis=1)
    y_balanced = df_balanced['target']
    
    return X_balanced, y_balanced

X_balanced, y_balanced = enhanced_balance_classes(X, y_encoded)
imputer_balanced = SimpleImputer(strategy='median')
X_balanced_clean = pd.DataFrame(imputer_balanced.fit_transform(X_balanced), columns=X_balanced.columns)
X_balanced_final = X_balanced_clean.values
y_balanced_final = y_balanced.values

# 8. Enhanced feature selection with multiple methods
# Method 1: Mutual information
selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(10, X_balanced_final.shape[1]))
X_selected_mi = selector_mi.fit_transform(X_balanced_final, y_balanced_final)
selected_features_mi = X_balanced.columns[selector_mi.get_support()]

# Method 2: Recursive Feature Elimination with XGBoost
rfe_selector = RFE(estimator=XGBClassifier(random_state=42, eval_metric='mlogloss'), n_features_to_select=10)
X_selected_rfe = rfe_selector.fit_transform(X_balanced_final, y_balanced_final)
selected_features_rfe = X_balanced.columns[rfe_selector.get_support()]

# Combine both methods - use intersection of important features
combined_features = set(selected_features_mi) & set(selected_features_rfe)
if len(combined_features) < 6:  # Ensure minimum features
    combined_features = set(selected_features_mi) | set(selected_features_rfe)

# Select final features
feature_indices = [i for i, col in enumerate(X_balanced.columns) if col in combined_features]
X_selected = X_balanced_final[:, feature_indices]
selected_features = [X_balanced.columns[i] for i in feature_indices]

print(f"Selected features: {selected_features}")

# 9. Scaling with StandardScaler for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# 10. Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_balanced_final, test_size=0.2, random_state=42, stratify=y_balanced_final
)

# === IMPROVED XGBOOST MODEL TRAINING ===
print("\n=== IMPROVED XGBOOST MODEL TRAINING ===")

# Enhanced parameter grid
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 1.5]
}

# Use more sophisticated cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initial XGBoost model with better default parameters
xgb_model = XGBClassifier(
    use_label_encoder=False, 
    eval_metric='mlogloss', 
    random_state=42,
    tree_method='hist',  # Faster training
    objective='multi:softprob'
)

# Grid search with better scoring
grid_search = GridSearchCV(
    xgb_model, 
    param_grid, 
    scoring='accuracy', 
    cv=cv, 
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_xgb = grid_search.best_estimator_

print(f"Best XGBoost parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate improved XGBoost
y_pred_xgb = best_xgb.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"\nImproved XGBoost Accuracy: {xgb_accuracy:.4f}")

# Training accuracy to check for overfitting
y_train_pred = best_xgb.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Overfitting Check (Train-Test diff): {train_accuracy - xgb_accuracy:.4f}")

print("\nClassification Report (Improved XGBoost):")
target_names = label_encoder.classes_
print(classification_report(y_test, y_pred_xgb, target_names=target_names))

# Confusion matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print("\nConfusion Matrix (Improved XGBoost):")
print(cm_xgb)

# Cross-validation scores
cv_scores = cross_val_score(best_xgb, X_scaled, y_balanced_final, cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance
importances = best_xgb.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance (Improved XGBoost):")
print(feature_importance)

# Visualizations
import os
os.makedirs('public/improved_xgboost_analysis', exist_ok=True)

# Feature importance plot
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance (Improved XGBoost)', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig('public/improved_xgboost_analysis/improved_xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Confusion matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Improved XGBoost', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('public/improved_xgboost_analysis/improved_xgboost_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Performance comparison plot
plt.figure(figsize=(10, 6))
performance_data = {
    'Metric': ['Training Accuracy', 'Test Accuracy', 'CV Mean Accuracy'],
    'Score': [train_accuracy, xgb_accuracy, cv_scores.mean()]
}
performance_df = pd.DataFrame(performance_data)
sns.barplot(data=performance_df, x='Metric', y='Score', palette='Set2')
plt.title('Model Performance Metrics', fontsize=16)
plt.ylabel('Accuracy Score', fontsize=12)
plt.ylim(0, 1)
for i, v in enumerate(performance_df['Score']):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('public/improved_xgboost_analysis/performance_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# Save improved model & artifacts
joblib.dump(best_xgb, 'public/improved_xgboost_analysis/improved_xgboost_model.pkl')
joblib.dump(scaler, 'public/improved_xgboost_analysis/improved_scaler.pkl')
joblib.dump(label_encoder, 'public/improved_xgboost_analysis/improved_label_encoder.pkl')
feature_importance.to_csv('public/improved_xgboost_analysis/improved_xgboost_feature_importance.csv', index=False)

# Save feature mapping for prediction function
feature_mapping = {
    'selected_features': selected_features,
    'feature_indices': feature_indices,
    'original_columns': X_balanced.columns.tolist()
}
joblib.dump(feature_mapping, 'public/improved_xgboost_analysis/feature_mapping.pkl')

# Enhanced summary
improved_summary = {
    'Dataset_Shape': df_processed.shape,
    'Selected_Features': selected_features,
    'Removed_Features': ['Low_Platelet'],
    'New_Features': ['WBC_Age_Interaction', 'Platelet_Age_Interaction', 'Combined_Test_Score', 'Log_WBC', 'Log_Platelet'],
    'Model_Performance': {
        'XGBoost_Test_Accuracy': xgb_accuracy,
        'XGBoost_Train_Accuracy': train_accuracy,
        'CV_Mean_Score': cv_scores.mean(),
        'CV_Std_Score': cv_scores.std(),
        'Best_CV_Score': grid_search.best_score_,
        'Overfitting_Check': train_accuracy - xgb_accuracy
    },
    'Best_Parameters': grid_search.best_params_,
    'Cancer_Types': target_names.tolist(),
    'Feature_Importance': feature_importance.to_dict('records'),
    'Improvement_Steps': [
        'Removed Low_Platelet feature (0 importance)',
        'Added interaction features',
        'Added log-transformed features',
        'Enhanced feature selection (MI + RFE)',
        'Improved class balancing',
        'Extended hyperparameter tuning',
        'Better regularization parameters'
    ]
}

import json
with open('public/improved_xgboost_analysis/improved_xgboost_analysis_summary.json', 'w') as f:
    json.dump(improved_summary, f, indent=2)

print("\n=== IMPROVED XGBOOST ENHANCEMENT COMPLETE ===")
print(f"Previous accuracy: 63.68%")
print(f"Improved accuracy: {xgb_accuracy:.2%}")
print(f"Improvement: {((xgb_accuracy - 0.6368) * 100):.2f} percentage points")
print(f"Selected features: {len(selected_features)}")
print(f"Cancer types: {target_names}")
print(f"Overfitting check: {train_accuracy - xgb_accuracy:.4f}")

# Enhanced prediction function for new data
def predict_cancer_type_improved(age, gender, wbc_count, platelet_count, bone_marrow, spep, lymph_node, lumbar_puncture, genetic_data):
    """
    Enhanced function to predict cancer type for new patient data using improved XGBoost
    """
    # Prepare input data with all original features
    input_data = {
        'Age': age,
        'Gender': gender,
        'Total WBC count(/cumm)': wbc_count,
        'Platelet Count( (/cumm)': platelet_count,
        'Bone Marrow Aspiration(Positive / Negative / Not Done)': bone_marrow,
        'Serum Protein Electrophoresis (SPEP)(Normal / Abnormal)': spep,
        'Lymph Node Biopsy(Positive / Negative / Not Done)': lymph_node,
        'Lumbar Puncture (Spinal Tap)': lumbar_puncture,
        'Genetic_Data(BCR-ABL, FLT3)': genetic_data
    }
    
    # Calculate engineered features
    input_data['WBC_Platelet_Ratio'] = wbc_count / (platelet_count + 1)
    input_data['Age_Group'] = 0 if age <= 30 else 1 if age <= 60 else 2
    input_data['High_WBC'] = 1 if wbc_count > X['Total WBC count(/cumm)'].quantile(0.75) else 0
    input_data['WBC_Age_Interaction'] = wbc_count * age
    input_data['Platelet_Age_Interaction'] = platelet_count * age
    input_data['Combined_Test_Score'] = bone_marrow + lymph_node + spep
    input_data['Log_WBC'] = np.log1p(wbc_count)
    input_data['Log_Platelet'] = np.log1p(platelet_count)
    
    # Create DataFrame and select features
    input_df = pd.DataFrame([input_data])
    input_selected = input_df[selected_features].values
    
    # Scale and predict
    input_scaled = scaler.transform(input_selected)
    prediction = best_xgb.predict(input_scaled)[0]
    probability = best_xgb.predict_proba(input_scaled)[0]
    
    predicted_cancer = label_encoder.inverse_transform([prediction])[0]
    confidence = max(probability)
    
    # Get probabilities for all classes
    prob_dict = {label_encoder.classes_[i]: prob for i, prob in enumerate(probability)}
    
    return predicted_cancer, confidence, prob_dict

print("\nImproved model ready for predictions!")
print("Use predict_cancer_type_improved() function for new predictions.")
print("This function now returns predicted cancer type, confidence, and probability distribution.")