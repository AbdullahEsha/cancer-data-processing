import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.utils import resample
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("=== OPTIMIZED XGBOOST FOR KEY FEATURES BLOOD CANCER CLASSIFICATION ===")

# Load the key features dataset
try:
    df = pd.read_csv('public/test_analysis/processed_blood_cancer_dataset.csv')
    print(f"✅ Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
except FileNotFoundError:
    print("❌ Error: Could not find the processed dataset file.")
    print("Please ensure 'public/test_analysis/processed_blood_cancer_dataset.csv' exists.")
    exit()

# Verify we have the expected key features
expected_columns = ["Platelet Count( (/cumm)", "Total WBC count(/cumm)", "Age", "Cancer_Type(AML, ALL, CLL)"]
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    print(f"❌ Missing expected columns: {missing_columns}")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

print(f"\n=== INITIAL DATA ANALYSIS ===")
print(f"Dataset shape: {df.shape}")
print(f"\nTarget variable distribution:")
print(df['Cancer_Type(AML, ALL, CLL)'].value_counts())
print(f"\nMissing values per column:")
print(df.isnull().sum())

# STEP 1: ENHANCED DATA PREPROCESSING
print(f"\n=== ENHANCED DATA PREPROCESSING ===")

# Create working copy
df_processed = df.copy()

# 1. Drop all null values for cleaner dataset
print(f"Rows before dropping nulls: {len(df_processed)}")
df_processed = df_processed.dropna()
print(f"Rows after dropping nulls: {len(df_processed)}")

if len(df_processed) == 0:
    print("❌ No data remaining after dropping nulls!")
    exit()

# 2. Data type conversion and validation
numeric_features = ["Platelet Count( (/cumm)", "Total WBC count(/cumm)", "Age"]
for col in numeric_features:
    # Convert to numeric, coercing any non-numeric values to NaN
    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

# Drop any rows where conversion failed
df_processed = df_processed.dropna()
print(f"Rows after numeric conversion: {len(df_processed)}")

# 3. Advanced outlier handling using IQR method with caps
print(f"\n=== OUTLIER HANDLING ===")
for col in numeric_features:
    Q1 = df_processed[col].quantile(0.25)
    Q3 = df_processed[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Set more conservative bounds (1.5 IQR rule)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count outliers before capping
    outliers_before = len(df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)])
    print(f"{col}: {outliers_before} outliers detected")
    
    # Cap outliers instead of removing them
    df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)

# 4. Filter main cancer types only
main_cancer_types = ['AML', 'ALL', 'CLL']
if df_processed['Cancer_Type(AML, ALL, CLL)'].dtype == 'object':
    df_processed = df_processed[df_processed['Cancer_Type(AML, ALL, CLL)'].isin(main_cancer_types)]
else:
    # If already encoded, check unique values
    unique_values = df_processed['Cancer_Type(AML, ALL, CLL)'].unique()
    print(f"Unique cancer type values: {unique_values}")

print(f"Dataset shape after filtering main cancer types: {df_processed.shape}")

# 5. ADVANCED FEATURE ENGINEERING
print(f"\n=== ADVANCED FEATURE ENGINEERING ===")

# Basic ratios and transformations
df_processed['WBC_Platelet_Ratio'] = df_processed['Total WBC count(/cumm)'] / (df_processed['Platelet Count( (/cumm)'] + 1)
df_processed['Platelet_WBC_Ratio'] = df_processed['Platelet Count( (/cumm)'] / (df_processed['Total WBC count(/cumm)'] + 1)

# Age-based features
df_processed['Age_Group'] = pd.cut(df_processed['Age'], bins=[0, 30, 60, 100], labels=[0, 1, 2])
df_processed['Age_Group'] = df_processed['Age_Group'].astype(int)
df_processed['Age_Squared'] = df_processed['Age'] ** 2
df_processed['Age_Sqrt'] = np.sqrt(df_processed['Age'])

# Statistical thresholds
wbc_75th = df_processed['Total WBC count(/cumm)'].quantile(0.75)
wbc_25th = df_processed['Total WBC count(/cumm)'].quantile(0.25)
platelet_75th = df_processed['Platelet Count( (/cumm)'].quantile(0.75)
platelet_25th = df_processed['Platelet Count( (/cumm)'].quantile(0.25)

df_processed['High_WBC'] = (df_processed['Total WBC count(/cumm)'] > wbc_75th).astype(int)
df_processed['Low_WBC'] = (df_processed['Total WBC count(/cumm)'] < wbc_25th).astype(int)
df_processed['High_Platelet'] = (df_processed['Platelet Count( (/cumm)'] > platelet_75th).astype(int)
df_processed['Low_Platelet'] = (df_processed['Platelet Count( (/cumm)'] < platelet_25th).astype(int)

# Interaction features
df_processed['WBC_Age_Interaction'] = df_processed['Total WBC count(/cumm)'] * df_processed['Age']
df_processed['Platelet_Age_Interaction'] = df_processed['Platelet Count( (/cumm)'] * df_processed['Age']
df_processed['WBC_Platelet_Product'] = df_processed['Total WBC count(/cumm)'] * df_processed['Platelet Count( (/cumm)']

# Log transformations for skewed distributions
df_processed['Log_WBC'] = np.log1p(df_processed['Total WBC count(/cumm)'])
df_processed['Log_Platelet'] = np.log1p(df_processed['Platelet Count( (/cumm)'])
df_processed['Log_Age'] = np.log1p(df_processed['Age'])

# Normalization features (z-scores)
df_processed['WBC_Zscore'] = (df_processed['Total WBC count(/cumm)'] - df_processed['Total WBC count(/cumm)'].mean()) / df_processed['Total WBC count(/cumm)'].std()
df_processed['Platelet_Zscore'] = (df_processed['Platelet Count( (/cumm)'] - df_processed['Platelet Count( (/cumm)'].mean()) / df_processed['Platelet Count( (/cumm)'].std()

# Combined risk scores
df_processed['Blood_Count_Risk_Score'] = (
    df_processed['High_WBC'] * 2 + 
    df_processed['Low_Platelet'] * 2 + 
    df_processed['Low_WBC'] * 1 + 
    df_processed['High_Platelet'] * 0.5
)

# Power transformations
df_processed['WBC_Power_2'] = df_processed['Total WBC count(/cumm)'] ** 2
df_processed['Platelet_Power_2'] = df_processed['Platelet Count( (/cumm)'] ** 2
df_processed['WBC_Sqrt'] = np.sqrt(df_processed['Total WBC count(/cumm)'])
df_processed['Platelet_Sqrt'] = np.sqrt(df_processed['Platelet Count( (/cumm)'])

print(f"Total features after engineering: {len(df_processed.columns)}")
print(f"New features created: {len(df_processed.columns) - 4}")

# 6. Prepare features and target
X = df_processed.drop('Cancer_Type(AML, ALL, CLL)', axis=1)
y = df_processed['Cancer_Type(AML, ALL, CLL)']

# Encode target variable if it's categorical
if y.dtype == 'object':
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    target_names = label_encoder.classes_
else:
    y_encoded = y.values
    target_names = np.unique(y_encoded)
    label_encoder = None

print(f"Feature matrix shape: {X.shape}")
print(f"Target classes: {target_names}")
print(f"Class distribution: {pd.Series(y_encoded).value_counts().sort_index()}")

# 7. INTELLIGENT CLASS BALANCING
print(f"\n=== INTELLIGENT CLASS BALANCING ===")

def enhanced_balance_classes(X, y):
    """Enhanced class balancing with noise injection"""
    df_combined = pd.concat([X.reset_index(drop=True), pd.Series(y, name='target').reset_index(drop=True)], axis=1)
    
    class_counts = df_combined['target'].value_counts()
    print(f"Original class distribution: {class_counts.to_dict()}")
    
    # Use median size with multiplier for target
    target_size = int(np.median(class_counts) * 1.3)  # 30% above median
    
    balanced_dfs = []
    for class_val in sorted(df_combined['target'].unique()):
        class_df = df_combined[df_combined['target'] == class_val]
        
        if len(class_df) < target_size:
            # Oversample with slight noise injection
            oversampled = resample(class_df, replace=True, n_samples=target_size, random_state=42)
            # Add small amount of noise to numeric features
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in oversampled.columns:
                    noise = np.random.normal(0, oversampled[col].std() * 0.01, len(oversampled))
                    oversampled[col] += noise
            balanced_dfs.append(oversampled)
        else:
            # Keep original if not too large
            balanced_dfs.append(class_df)
    
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X_balanced = df_balanced.drop('target', axis=1)
    y_balanced = df_balanced['target'].values
    
    balanced_counts = pd.Series(y_balanced).value_counts()
    print(f"Balanced class distribution: {balanced_counts.to_dict()}")
    
    return X_balanced, y_balanced

X_balanced, y_balanced = enhanced_balance_classes(X, y_encoded)

# 8. ADVANCED FEATURE SELECTION
print(f"\n=== ADVANCED FEATURE SELECTION ===")

# Method 1: Mutual Information
selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(15, X_balanced.shape[1]))
X_selected_mi = selector_mi.fit_transform(X_balanced, y_balanced)
selected_features_mi = X_balanced.columns[selector_mi.get_support()]

# Method 2: F-statistics
selector_f = SelectKBest(score_func=f_classif, k=min(15, X_balanced.shape[1]))
X_selected_f = selector_f.fit_transform(X_balanced, y_balanced)
selected_features_f = X_balanced.columns[selector_f.get_support()]

# Method 3: XGBoost feature importance
temp_xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
temp_xgb.fit(X_balanced, y_balanced)
importance_scores = temp_xgb.feature_importances_
top_features_xgb = X_balanced.columns[np.argsort(importance_scores)[-15:][::-1]]

# Combine feature selection methods
combined_features = set(selected_features_mi) | set(selected_features_f) | set(top_features_xgb)
combined_features = list(combined_features)

print(f"Features selected by Mutual Information: {len(selected_features_mi)}")
print(f"Features selected by F-statistics: {len(selected_features_f)}")
print(f"Features selected by XGBoost importance: {len(top_features_xgb)}")
print(f"Combined unique features: {len(combined_features)}")
print(f"Selected features: {combined_features}")

# Use combined features
X_selected = X_balanced[combined_features]

# 9. ROBUST SCALING
print(f"\n=== ROBUST SCALING ===")
scaler = RobustScaler()  # More robust to outliers than StandardScaler
X_scaled = scaler.fit_transform(X_selected)

# 10. STRATIFIED TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# === OPTIMIZED TWO-STAGE XGBOOST TRAINING ===
print(f"\n=== OPTIMIZED TWO-STAGE XGBOOST TRAINING ===")

# Stage 1: Quick parameter screening
print("Stage 1: Quick parameter screening...")
quick_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

cv_quick = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

xgb_model_quick = XGBClassifier(
    eval_metric='mlogloss',
    random_state=42,
    tree_method='hist',
    objective='multi:softprob',
    n_jobs=1
)

quick_grid_search = GridSearchCV(
    xgb_model_quick,
    quick_param_grid,
    scoring='accuracy',
    cv=cv_quick,
    n_jobs=-1,
    verbose=1
)

quick_grid_search.fit(X_train, y_train)
print(f"Quick search best params: {quick_grid_search.best_params_}")
print(f"Quick search best score: {quick_grid_search.best_score_:.4f}")

# Stage 2: Fine-tuning around best parameters
print("\nStage 2: Fine-tuning best parameters...")
best_params = quick_grid_search.best_params_

refined_param_grid = {
    'n_estimators': [best_params['n_estimators'], min(500, best_params['n_estimators'] + 100)],
    'max_depth': [best_params['max_depth']],
    'learning_rate': [best_params['learning_rate']],
    'subsample': [best_params['subsample']],
    'colsample_bytree': [best_params['colsample_bytree']],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2],
    'min_child_weight': [1, 3, 5]
}

cv_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_model_final = XGBClassifier(
    eval_metric='mlogloss',
    random_state=42,
    tree_method='hist',
    objective='multi:softprob',
    n_jobs=1
)

final_grid_search = GridSearchCV(
    xgb_model_final,
    refined_param_grid,
    scoring='accuracy',
    cv=cv_final,
    n_jobs=-1,
    verbose=1
)

final_grid_search.fit(X_train, y_train)
best_xgb = final_grid_search.best_estimator_

print(f"Final best params: {final_grid_search.best_params_}")
print(f"Final best CV score: {final_grid_search.best_score_:.4f}")

# Stage 3: Ensemble approach
print("\nStage 3: Creating ensemble model...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

ensemble_model = VotingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('rf', rf_model),
        ('gb', gb_model)
    ],
    voting='soft',
    n_jobs=-1
)

print("Training ensemble model...")
ensemble_model.fit(X_train, y_train)

# === MODEL EVALUATION ===
print(f"\n=== MODEL EVALUATION ===")

# Compare models
xgb_pred = best_xgb.predict(X_test)
ensemble_pred = ensemble_model.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_pred)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')

print(f"XGBoost - Accuracy: {xgb_accuracy:.4f}, F1: {xgb_f1:.4f}")
print(f"Ensemble - Accuracy: {ensemble_accuracy:.4f}, F1: {ensemble_f1:.4f}")

# Select best model
if ensemble_accuracy > xgb_accuracy:
    print("🏆 Using ensemble model (better performance)")
    best_model = ensemble_model
    y_pred_final = ensemble_pred
    final_accuracy = ensemble_accuracy
    final_f1 = ensemble_f1
    model_type = "Ensemble"
else:
    print("🏆 Using XGBoost model (better performance)")
    best_model = best_xgb
    y_pred_final = xgb_pred
    final_accuracy = xgb_accuracy
    final_f1 = xgb_f1
    model_type = "XGBoost"

# Training accuracy check
y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

print(f"\n🎯 FINAL RESULTS:")
print(f"Model Type: {model_type}")
print(f"Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
print(f"Test F1-Score: {final_f1:.4f}")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Overfitting Check: {train_accuracy - final_accuracy:.4f}")

# Cross-validation
cv_scores = cross_val_score(best_model, X_scaled, y_balanced, cv=5, scoring='accuracy')
print(f"CV Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Detailed classification report
print(f"\n📊 Classification Report:")
if label_encoder is not None:
    print(classification_report(y_test, y_pred_final, target_names=target_names))
else:
    print(classification_report(y_test, y_pred_final))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_final)
print(f"\n📊 Confusion Matrix:")
print(cm)

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': combined_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
elif hasattr(best_model, 'named_estimators_'):
    # For ensemble, get XGBoost importance
    xgb_estimator = best_model.named_estimators_['xgb']
    feature_importance = pd.DataFrame({
        'Feature': combined_features,
        'Importance': xgb_estimator.feature_importances_
    }).sort_values('Importance', ascending=False)

print(f"\n📊 Top 10 Feature Importances:")
print(feature_importance.head(10))

# === SAVE MODELS AND RESULTS ===
print(f"\n💾 SAVING MODELS AND RESULTS...")

# Create output directory
os.makedirs('public/optimized_key_features_analysis', exist_ok=True)

# Save models
if model_type == "Ensemble":
    joblib.dump(best_model, 'public/optimized_key_features_analysis/ensemble_model.pkl')
    joblib.dump(best_xgb, 'public/optimized_key_features_analysis/xgboost_model.pkl')
else:
    joblib.dump(best_model, 'public/optimized_key_features_analysis/xgboost_model.pkl')

# Save preprocessing components
joblib.dump(scaler, 'public/optimized_key_features_analysis/scaler.pkl')
if label_encoder is not None:
    joblib.dump(label_encoder, 'public/optimized_key_features_analysis/label_encoder.pkl')

# Save feature information
feature_info = {
    'selected_features': combined_features,
    'original_features': ["Platelet Count( (/cumm)", "Total WBC count(/cumm)", "Age"],
    'engineered_features': list(set(combined_features) - set(["Platelet Count( (/cumm)", "Total WBC count(/cumm)", "Age"]))
}
joblib.dump(feature_info, 'public/optimized_key_features_analysis/feature_info.pkl')

# Save feature importance
feature_importance.to_csv('public/optimized_key_features_analysis/feature_importance.csv', index=False)

# === VISUALIZATIONS ===
print(f"\n📊 CREATING VISUALIZATIONS...")

# 1. Feature Importance Plot
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
sns.barplot(data=top_features, x='Importance', y='Feature', palette='viridis')
plt.title(f'Top 15 Feature Importances - {model_type} Model\nAccuracy: {final_accuracy*100:.2f}%', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.tight_layout()
plt.savefig('public/optimized_key_features_analysis/feature_importance_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Confusion Matrix
plt.figure(figsize=(10, 8))
if label_encoder is not None:
    labels = target_names
else:
    labels = [f'Class {i}' for i in range(len(np.unique(y_test)))]

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title(f'Confusion Matrix - {model_type} Model\nAccuracy: {final_accuracy*100:.2f}%', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('public/optimized_key_features_analysis/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Performance Metrics
plt.figure(figsize=(12, 6))
metrics_data = {
    'Metric': ['Training Accuracy', 'Test Accuracy', 'CV Mean Accuracy', 'F1-Score'],
    'Score': [train_accuracy, final_accuracy, cv_scores.mean(), final_f1]
}
metrics_df = pd.DataFrame(metrics_data)

sns.barplot(data=metrics_df, x='Metric', y='Score', palette='Set2')
plt.title(f'Model Performance Metrics - {model_type}', fontsize=16)
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1)
for i, v in enumerate(metrics_df['Score']):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('public/optimized_key_features_analysis/performance_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Cross-validation scores
plt.figure(figsize=(10, 6))
plt.bar(range(len(cv_scores)), cv_scores, alpha=0.7)
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
plt.title('Cross-Validation Accuracy Scores', fontsize=16)
plt.xlabel('Fold', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('public/optimized_key_features_analysis/cv_scores.png', dpi=300, bbox_inches='tight')
plt.show()

# === SUMMARY REPORT ===
summary_report = {
    'Dataset_Info': {
        'Original_Shape': df.shape,
        'Final_Shape': df_processed.shape,
        'Rows_Dropped_Nulls': df.shape[0] - len(df_processed),
        'Features_Engineered': len(df_processed.columns) - 4,
        'Selected_Features': len(combined_features)
    },
    'Model_Performance': {
        'Model_Type': model_type,
        'Test_Accuracy': float(final_accuracy),
        'Test_F1_Score': float(final_f1),
        'Training_Accuracy': float(train_accuracy),
        'CV_Mean_Accuracy': float(cv_scores.mean()),
        'CV_Std_Accuracy': float(cv_scores.std()),
        'Overfitting_Check': float(train_accuracy - final_accuracy),
        'XGBoost_Accuracy': float(xgb_accuracy),
        'Ensemble_Accuracy': float(ensemble_accuracy) if 'ensemble_accuracy' in locals() else None
    },
    'Best_Parameters': final_grid_search.best_params_,
    'Feature_Engineering': {
        'Original_Features': ["Platelet Count( (/cumm)", "Total WBC count(/cumm)", "Age"],
        'Engineered_Features': list(set(combined_features) - set(["Platelet Count( (/cumm)", "Total WBC count(/cumm)", "Age"]))[:10],  # Top 10
        'Feature_Selection_Methods': ['Mutual Information', 'F-Statistics', 'XGBoost Importance']
    },
    'Top_10_Features': feature_importance.head(10)[['Feature', 'Importance']].to_dict('records'),
    'Cancer_Types': target_names.tolist() if label_encoder is not None else list(np.unique(y_encoded)),
    'Optimization_Techniques': [
        'Two-stage hyperparameter tuning',
        'Ensemble voting classifier',
        'Advanced feature engineering',
        'Intelligent class balancing',
        'Robust scaling',
        'Multiple feature selection methods',
        'Outlier capping with IQR',
        'Noise injection for oversampling'
    ]
}

# Save summary
with open('public/optimized_key_features_analysis/analysis_summary.json', 'w') as f:
    json.dump(summary_report, f, indent=2)

# === PREDICTION FUNCTION ===
def predict_cancer_type(features):
    """Predict cancer type given a feature vector."""
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame([features], columns=X_selected.columns)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict using the best model
    prediction = best_model.predict(features_scaled)
    
    if label_encoder is not None:
        prediction = label_encoder.inverse_transform(prediction)
    
    return prediction[0]

# Example usage
example_features = {
    'Platelet Count( (/cumm)': 200000,
    'Total WBC count(/cumm)': 8000,
    'Age': 45,
    'WBC_Platelet_Ratio': 0.04,
    'Platelet_WBC_Ratio': 25.0,
    'Age_Group': 1,
    'Age_Squared': 2025,
    'Age_Sqrt': 6.708,
    'High_WBC': 0,
    'Low_WBC': 0,
    'High_Platelet': 0, 
    'Low_Platelet': 0,
    'WBC_Age_Interaction': 360000,
    'Platelet_Age_Interaction': 9000000,
    'WBC_Platelet_Product': 1600000000,
    'Log_WBC': 8.987,
    'Log_Platelet': 12.206,
    'Log_Age': 3.806,
    'WBC_Zscore': 0.0,
    'Platelet_Zscore': 0.0,
    'Blood_Count_Risk_Score': 0,
    'WBC_Power_2': 64000000,
    'Platelet_Power_2': 40000000000,
    'WBC_Sqrt': 89.442,
    'Platelet_Sqrt': 447.214
}

predicted_cancer_type = predict_cancer_type(example_features)
print(f"\n🔮 Predicted Cancer Type: {predicted_cancer_type}")
print("💾 All results saved successfully in 'public/optimized_key_features_analysis/' directory.")

metrics = {
    'accuracy': final_accuracy,
    'f1_score': final_f1,
    'train_accuracy': train_accuracy,
    'cv_mean_accuracy': cv_scores.mean(),
    'cv_std_accuracy': cv_scores.std(),
    'best_params': final_grid_search.best_params_,
    'model_type': model_type,
    'feature_importance': feature_importance.to_dict(orient='records'),
    'selected_features': combined_features,
    'cancer_types': target_names.tolist() if label_encoder is not None else list(np.unique(y_balanced)),
    'summary_report': summary_report
}

metrics_file = 'public/optimized_key_features_analysis/model_metrics.json'
with open(metrics_file, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✅ Model metrics saved to: {metrics_file}")
print(f"✅ All results saved successfully in 'public/optimized_key_features_analysis/' directory.")
print("🎉 Analysis complete! Optimized XGBoost model for blood cancer classification is ready "
      "for deployment and further analysis.")
print("Thank you for using the Optimized XGBoost Key Features Analysis script!")