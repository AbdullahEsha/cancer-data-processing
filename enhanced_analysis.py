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
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('public/blood_cancer_diseases_dataset.csv')

print("=== ENHANCED BLOOD CANCER CLASSIFICATION MODEL ===")
print(f"Original dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# STEP 1: ADVANCED DATA PREPROCESSING
print("\n=== ADVANCED DATA PREPROCESSING ===")

# Create a copy for preprocessing
df_processed = df.copy()

# 1. Handle target variable first - filter out only the main cancer types
main_cancer_types = ['AML', 'ALL', 'CLL']
df_processed = df_processed[df_processed['Cancer_Type(AML, ALL, CLL)'].isin(main_cancer_types)]
print(f"Filtered dataset shape (main cancer types only): {df_processed.shape}")

# Check class distribution
print("\nClass distribution:")
print(df_processed['Cancer_Type(AML, ALL, CLL)'].value_counts())

# 2. Intelligent feature selection - Remove potentially problematic columns
# Remove columns that might cause data leakage or are not predictive
columns_to_drop = [
    'Comments',  # Too generic
    'Diagnosis_Result',  # This might be post-diagnosis information
    'Treatment_Outcome',  # This is outcome, not predictor
    'Treatment_Type(Chemotherapy, Radiation)',  # Treatment is chosen after diagnosis
    'Side_Effects'  # Side effects happen after treatment
]

# Keep only relevant predictive features
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
print(f"Dataset shape after feature selection: {df_processed.shape}")

# 3. Advanced missing value handling
print("\nMissing values before preprocessing:")
print(df_processed.isnull().sum())

# Handle numeric features
numeric_features = ['Age', 'Total WBC count(/cumm)', 'Platelet Count( (/cumm)']
for col in numeric_features:
    if col in df_processed.columns:
        # Convert to numeric and handle outliers
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Remove extreme outliers using IQR method
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing them
        df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)
        
        # Fill missing values with median
        df_processed[col].fillna(df_processed[col].median(), inplace=True)

# 4. Improved categorical encoding
# Gender encoding
df_processed['Gender'] = df_processed['Gender'].map({'Male': 0, 'Female': 1})

# Bone Marrow Aspiration - more meaningful encoding
df_processed['Bone Marrow Aspiration(Positive / Negative / Not Done)'] = df_processed['Bone Marrow Aspiration(Positive / Negative / Not Done)'].map({
    'Positive': 2,    # Most indicative of cancer
    'Negative': 0,    # Least indicative
    'Not Done': 1     # Intermediate (unknown)
})

# SPEP encoding
df_processed['Serum Protein Electrophoresis (SPEP)(Normal / Abnormal)'] = df_processed['Serum Protein Electrophoresis (SPEP)(Normal / Abnormal)'].map({
    'Normal': 0,
    'Abnormal': 1,
    'Not Done': 0.5  # Intermediate value
})

# Lymph Node Biopsy
df_processed['Lymph Node Biopsy(Positive / Negative / Not Done)'] = df_processed['Lymph Node Biopsy(Positive / Negative / Not Done)'].map({
    'Positive': 2,
    'Negative': 0,
    'Not Done': 1
})

# Lumbar Puncture - encode based on cancer type relevance
lumbar_puncture_mapping = {}
for i, val in enumerate(df_processed['Lumbar Puncture (Spinal Tap)'].unique()):
    lumbar_puncture_mapping[val] = i
df_processed['Lumbar Puncture (Spinal Tap)'] = df_processed['Lumbar Puncture (Spinal Tap)'].map(lumbar_puncture_mapping)

# Genetic Data - this is very important for cancer classification
genetic_data_mapping = {}
for i, val in enumerate(df_processed['Genetic_Data(BCR-ABL, FLT3)'].unique()):
    genetic_data_mapping[val] = i
df_processed['Genetic_Data(BCR-ABL, FLT3)'] = df_processed['Genetic_Data(BCR-ABL, FLT3)'].map(genetic_data_mapping)

# Fill any remaining missing values for numeric columns only
numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df_processed[col].fillna(df_processed[col].median(), inplace=True)

# Fill categorical columns with mode
categorical_cols = df_processed.select_dtypes(include=['object']).columns
for col in categorical_cols:
    mode_value = df_processed[col].mode()
    if len(mode_value) > 0:
        df_processed[col].fillna(mode_value[0], inplace=True)
    else:
        df_processed[col].fillna('Unknown', inplace=True)

print("Missing values after preprocessing:")
print(df_processed.isnull().sum())

# 5. Feature Engineering - Create new meaningful features
df_processed['WBC_Platelet_Ratio'] = df_processed['Total WBC count(/cumm)'] / (df_processed['Platelet Count( (/cumm)'] + 1)
df_processed['Age_Group'] = pd.cut(df_processed['Age'], bins=[0, 30, 60, 100], labels=[0, 1, 2])
df_processed['High_WBC'] = (df_processed['Total WBC count(/cumm)'] > df_processed['Total WBC count(/cumm)'].quantile(0.75)).astype(int)
df_processed['Low_Platelet'] = (df_processed['Platelet Count( (/cumm)'] < df_processed['Platelet Count( (/cumm)'].quantile(0.25)).astype(int)

# Convert Age_Group to numeric
df_processed['Age_Group'] = df_processed['Age_Group'].astype(int)

# 6. Prepare features and target
X = df_processed.drop('Cancer_Type(AML, ALL, CLL)', axis=1)
y = df_processed['Cancer_Type(AML, ALL, CLL)']

# Ensure all features are numeric and handle any remaining NaN values
print(f"Data types before final processing:")
print(X.dtypes)
print(f"\nMissing values in features:")
print(X.isnull().sum())

# Convert any remaining object columns to numeric
for col in X.columns:
    if X[col].dtype == 'object':
        # Try to convert to numeric, if fails, use label encoding
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

# Final imputation for any remaining NaN values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

# Verify no NaN values remain
print(f"\nFinal check - Missing values after imputation:")
print(X_imputed.isnull().sum())
print(f"Any NaN values remaining: {X_imputed.isnull().any().any()}")

# Use the imputed data
X = X_imputed

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Target distribution: {np.bincount(y_encoded)}")

# 7. Handle class imbalance using SMOTE-like approach
def balance_classes(X, y):
    """Simple oversampling for minority classes"""
    # Ensure X is a DataFrame and y is a Series
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    # Reset indices to avoid issues
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    df_balanced = pd.concat([X, y.rename('target')], axis=1)
    
    # Get unique classes
    unique_classes = sorted(df_balanced['target'].unique())
    print(f"Unique classes: {unique_classes}")
    
    # Separate classes
    class_dfs = []
    for cls in unique_classes:
        class_df = df_balanced[df_balanced['target'] == cls]
        class_dfs.append(class_df)
        print(f"Class {cls} size: {len(class_df)}")
    
    # Find the maximum class size
    max_size = max(len(df) for df in class_dfs)
    print(f"Target size for balancing: {max_size}")
    
    # Oversample minority classes
    balanced_dfs = []
    for i, class_df in enumerate(class_dfs):
        if len(class_df) < max_size:
            # Oversample
            oversampled = resample(class_df, replace=True, n_samples=max_size, random_state=42)
            balanced_dfs.append(oversampled)
            print(f"Class {unique_classes[i]} oversampled from {len(class_df)} to {len(oversampled)}")
        else:
            balanced_dfs.append(class_df)
            print(f"Class {unique_classes[i]} kept at {len(class_df)}")
    
    # Combine balanced classes
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle the data
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X_balanced = df_balanced.drop('target', axis=1)
    y_balanced = df_balanced['target']
    
    # Final check for NaN values
    print(f"NaN values in balanced X: {X_balanced.isnull().sum().sum()}")
    print(f"NaN values in balanced y: {y_balanced.isnull().sum()}")
    
    return X_balanced, y_balanced

X_balanced, y_balanced = balance_classes(X, y_encoded)
print(f"Balanced dataset shape: {X_balanced.shape}")
print(f"Balanced target distribution: {np.bincount(y_balanced)}")

# Additional safety check - remove any rows with NaN values
if X_balanced.isnull().any().any():
    print("Warning: Found NaN values after balancing, removing them...")
    mask = ~X_balanced.isnull().any(axis=1)
    X_balanced = X_balanced[mask]
    y_balanced = y_balanced[mask]
    print(f"Dataset shape after removing NaN rows: {X_balanced.shape}")

# Final imputation as safety measure
imputer_balanced = SimpleImputer(strategy='median')
X_balanced_clean = pd.DataFrame(
    imputer_balanced.fit_transform(X_balanced), 
    columns=X_balanced.columns
)

# Convert to numpy arrays for sklearn
X_balanced_final = X_balanced_clean.values
y_balanced_final = y_balanced.values

print(f"Final balanced dataset shape: {X_balanced_final.shape}")
print(f"Final NaN check: {np.isnan(X_balanced_final).any()}")

# 8. Advanced feature selection
print("\n=== ADVANCED FEATURE SELECTION ===")

# Use mutual information for feature selection (better for classification)
selector = SelectKBest(score_func=mutual_info_classif, k=min(8, X_balanced_final.shape[1]))
X_selected = selector.fit_transform(X_balanced_final, y_balanced_final)
selected_features = X.columns[selector.get_support()]
print(f"Selected features: {selected_features.tolist()}")

# 9. Advanced scaling
scaler = RobustScaler()  # More robust to outliers than StandardScaler
X_scaled = scaler.fit_transform(X_selected)

# 10. Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_balanced_final, test_size=0.2, random_state=42, stratify=y_balanced_final
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# STEP 2: ENSEMBLE MODEL TRAINING
print("\n=== ENSEMBLE MODEL TRAINING ===")

# Define multiple models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42, class_weight='balanced', probability=True),
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
}

# Train and evaluate individual models
individual_scores = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    if name == 'SVM':
        # Grid search for SVM
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best SVM parameters: {grid_search.best_params_}")
    else:
        best_model = model
        best_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    individual_scores[name] = accuracy
    trained_models[name] = best_model
    
    print(f"{name} Accuracy: {accuracy:.4f}")

# Create ensemble model
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', trained_models['Random Forest']),
        ('gb', trained_models['Gradient Boosting']),
        ('svm', trained_models['SVM']),
        ('lr', trained_models['Logistic Regression'])
    ],
    voting='soft'
)

# Train ensemble
ensemble_model.fit(X_train, y_train)

# STEP 3: COMPREHENSIVE EVALUATION
print("\n=== COMPREHENSIVE EVALUATION ===")

# Evaluate ensemble model
y_pred_ensemble = ensemble_model.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

print(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")

# Detailed classification report
print("\nClassification Report (Ensemble):")
target_names = label_encoder.classes_
print(classification_report(y_test, y_pred_ensemble, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_ensemble)
print("\nConfusion Matrix:")
print(cm)

# Cross-validation
cv_scores = cross_val_score(ensemble_model, X_scaled, y_balanced_final, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance from Random Forest
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': trained_models['Random Forest'].feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance (Random Forest):")
print(feature_importance)

# STEP 4: VISUALIZATION
print("\n=== CREATING VISUALIZATIONS ===")

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance Analysis', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig('public/enhanced_svm_analysis/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Ensemble Model', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('public/enhanced_svm_analysis/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Model comparison
plt.figure(figsize=(12, 6))
model_names = list(individual_scores.keys()) + ['Ensemble']
model_scores = list(individual_scores.values()) + [ensemble_accuracy]

bars = plt.bar(model_names, model_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
plt.title('Model Performance Comparison', fontsize=16)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1)

# Add value labels on bars
for bar, score in zip(bars, model_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom', fontsize=11)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('public/enhanced_svm_analysis/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# STEP 5: SAVE ENHANCED MODEL
print("\n=== SAVING ENHANCED MODEL ===")

# Create directory if it doesn't exist
import os
os.makedirs('public/enhanced_svm_analysis', exist_ok=True)

# Save models and preprocessors
joblib.dump(ensemble_model, 'public/enhanced_svm_analysis/ensemble_model.pkl')
joblib.dump(scaler, 'public/enhanced_svm_analysis/scaler.pkl')
joblib.dump(selector, 'public/enhanced_svm_analysis/feature_selector.pkl')
joblib.dump(label_encoder, 'public/enhanced_svm_analysis/label_encoder.pkl')

# Save feature importance
feature_importance.to_csv('public/enhanced_svm_analysis/feature_importance.csv', index=False)

# Save enhanced summary
enhanced_summary = {
    'Dataset_Shape': df_processed.shape,
    'Selected_Features': selected_features.tolist(),
    'Model_Performance': {
        'Ensemble_Accuracy': ensemble_accuracy,
        'Individual_Model_Scores': individual_scores,
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
with open('public/enhanced_svm_analysis/enhanced_analysis_summary.json', 'w') as f:
    json.dump(enhanced_summary, f, indent=2)

print("\n=== ENHANCEMENT COMPLETE ===")
print(f"Original model accuracy: 18.08%")
print(f"Enhanced ensemble accuracy: {ensemble_accuracy:.2%}")
print(f"Improvement: {(ensemble_accuracy - 0.1808) / 0.1808 * 100:.1f}%")
print(f"Selected features: {len(selected_features)}")
print(f"Cancer types: {target_names}")

# Prediction function for new data
def predict_cancer_type(age, gender, wbc_count, platelet_count, bone_marrow, spep, lymph_node, lumbar_puncture, genetic_data):
    """
    Function to predict cancer type for new patient data
    """
    # Prepare input data
    input_data = np.array([[age, gender, wbc_count, platelet_count, bone_marrow, spep, lymph_node, lumbar_puncture, genetic_data]])
    
    # Add engineered features
    wbc_platelet_ratio = wbc_count / (platelet_count + 1)
    age_group = 0 if age <= 30 else 1 if age <= 60 else 2
    high_wbc = 1 if wbc_count > X['Total WBC count(/cumm)'].quantile(0.75) else 0
    low_platelet = 1 if platelet_count < X['Platelet Count( (/cumm)'].quantile(0.25) else 0
    
    # Extend input data with engineered features
    input_data = np.append(input_data, [[wbc_platelet_ratio, age_group, high_wbc, low_platelet]], axis=1)
    
    # Select features and scale
    input_selected = selector.transform(input_data)
    input_scaled = scaler.transform(input_selected)
    
    # Make prediction
    prediction = ensemble_model.predict(input_scaled)[0]
    probability = ensemble_model.predict_proba(input_scaled)[0]
    
    # Return result
    predicted_cancer = label_encoder.inverse_transform([prediction])[0]
    confidence = max(probability)
    
    return predicted_cancer, confidence

print("\nModel ready for predictions!")
print("Use predict_cancer_type() function for new predictions.")