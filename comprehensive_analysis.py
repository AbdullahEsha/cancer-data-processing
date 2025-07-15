import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('public/blood_cancer_diseases_dataset.csv')

print("=== Dataset Analysis ===")
print(f"Dataset shape: {df.shape}")
print(f"Cancer type distribution:")
print(df['Cancer_Type(AML, ALL, CLL)'].value_counts())
print(f"Percentage distribution:")
print(df['Cancer_Type(AML, ALL, CLL)'].value_counts(normalize=True) * 100)

# Data Preprocessing
print("\n=== ENHANCED DATA PREPROCESSING ===")

# Handle missing values
numeric_columns = ['Age', 'Total WBC count(/cumm)', 'Platelet Count( (/cumm)']
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)

categorical_columns = ['Genetic_Data(BCR-ABL, FLT3)', 'Side_Effects']
for col in categorical_columns:
    if col in df.columns:
        mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        df[col].fillna(mode_value, inplace=True)

# Encoding with better strategy
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Treatment types
treatment_types = df['Treatment_Type(Chemotherapy, Radiation)'].unique()
treatment_mapping = {treatment: idx for idx, treatment in enumerate(treatment_types)}
df['Treatment_Type(Chemotherapy, Radiation)'] = df['Treatment_Type(Chemotherapy, Radiation)'].map(treatment_mapping)

# Bone marrow aspiration
df['Bone Marrow Aspiration(Positive / Negative / Not Done)'] = df['Bone Marrow Aspiration(Positive / Negative / Not Done)'].map({
    'Positive': 1, 'Negative': 0, 'Not Done': -1
})

# SPEP
df['Serum Protein Electrophoresis (SPEP)(Normal / Abnormal)'] = df['Serum Protein Electrophoresis (SPEP)(Normal / Abnormal)'].map({
    'Normal': 0, 'Abnormal': 1, 'Not Done': -1
})

# Lymph node biopsy
df['Lymph Node Biopsy(Positive / Negative / Not Done)'] = df['Lymph Node Biopsy(Positive / Negative / Not Done)'].map({
    'Positive': 1, 'Negative': 0, 'Not Done': -1
})

# Lumbar puncture
lumbar_puncture_values = df['Lumbar Puncture (Spinal Tap)'].unique()
lumbar_puncture_mapping = {val: idx for idx, val in enumerate(lumbar_puncture_values)}
df['Lumbar Puncture (Spinal Tap)'] = df['Lumbar Puncture (Spinal Tap)'].map(lumbar_puncture_mapping)

# Genetic data
genetic_data_values = df['Genetic_Data(BCR-ABL, FLT3)'].unique()
genetic_data_mapping = {val: idx for idx, val in enumerate(genetic_data_values)}
df['Genetic_Data(BCR-ABL, FLT3)'] = df['Genetic_Data(BCR-ABL, FLT3)'].map(genetic_data_mapping)

# Side effects
side_effects_values = df['Side_Effects'].unique()
side_effects_mapping = {val: idx for idx, val in enumerate(side_effects_values)}
df['Side_Effects'] = df['Side_Effects'].map(side_effects_mapping)

# Comments
comments_values = df['Comments'].unique()
comments_mapping = {val: idx for idx, val in enumerate(comments_values)}
df['Comments'] = df['Comments'].map(comments_mapping)

# Cancer types (target)
cancer_types = df['Cancer_Type(AML, ALL, CLL)'].unique()
cancer_mapping = {cancer: idx for idx, cancer in enumerate(cancer_types)}
df['Cancer_Type(AML, ALL, CLL)'] = df['Cancer_Type(AML, ALL, CLL)'].map(cancer_mapping)

# Remove potential data leakage features
print("Removing potential data leakage features...")
leakage_features = ['Treatment_Outcome', 'Diagnosis_Result']
available_leakage = [col for col in leakage_features if col in df.columns]
if available_leakage:
    print(f"Removing: {available_leakage}")

# Remove rows with missing values
df.dropna(inplace=True)
print(f"Dataset shape after preprocessing: {df.shape}")

# Feature engineering
print("\n=== FEATURE ENGINEERING ===")
# Create new features
df['WBC_Platelet_Ratio'] = df['Total WBC count(/cumm)'] / (df['Platelet Count( (/cumm)'] + 1)
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, 100], labels=[0, 1, 2, 3, 4])
df['Age_Group'] = df['Age_Group'].astype(int)

# High/Low WBC and Platelet indicators
df['High_WBC'] = (df['Total WBC count(/cumm)'] > df['Total WBC count(/cumm)'].quantile(0.75)).astype(int)
df['Low_Platelet'] = (df['Platelet Count( (/cumm)'] < df['Platelet Count( (/cumm)'].quantile(0.25)).astype(int)

# Prepare features (exclude leakage features)
exclude_features = ['Cancer_Type(AML, ALL, CLL)'] + available_leakage
X = df.drop(columns=exclude_features)
y = df['Cancer_Type(AML, ALL, CLL)']

print(f"Feature matrix shape: {X.shape}")
print(f"Features used: {X.columns.tolist()}")

# Feature selection with multiple methods
print("\n=== ADVANCED FEATURE SELECTION ===")

# Method 1: Statistical selection
selector_statistical = SelectKBest(score_func=f_classif, k=min(15, X.shape[1]))
X_statistical = selector_statistical.fit_transform(X, y)
selected_features_statistical = X.columns[selector_statistical.get_support()]
print(f"Statistical selection features: {selected_features_statistical.tolist()}")

# Method 2: Tree-based selection
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
selector_tree = SelectFromModel(rf_selector, threshold='median')
X_tree = selector_tree.fit_transform(X, y)
selected_features_tree = X.columns[selector_tree.get_support()]
print(f"Tree-based selection features: {selected_features_tree.tolist()}")

# Use statistical selection for main analysis
X_selected = X_statistical
selected_features = selected_features_statistical

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")

# Handle class imbalance with SMOTE
print("\n=== HANDLING CLASS IMBALANCE ===")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"Balanced train shape: {X_train_balanced.shape}")
print(f"Balanced class distribution: {np.bincount(y_train_balanced)}")

# Scaling
scaler = RobustScaler()  # More robust to outliers than StandardScaler
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Define multiple algorithms
print("\n=== MULTIPLE ALGORITHM COMPARISON ===")

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, 
        max_depth=None, 
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        class_weight='balanced',
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=7,
        weights='distance'
    ),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    )
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate all models
results = {}
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train_balanced, 
                               cv=cv, scoring='accuracy')
    cv_f1 = cross_val_score(model, X_train_scaled, y_train_balanced, 
                           cv=cv, scoring='f1_weighted')
    
    # Fit and predict
    model.fit(X_train_scaled, y_train_balanced)
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {
        'model': model,
        'cv_accuracy': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'cv_f1': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
        'test_accuracy': accuracy,
        'test_f1': f1,
        'predictions': y_pred
    }
    
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"CV F1-Score: {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1-Score: {f1:.4f}")

# Find best model
best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

print(f"\n=== BEST MODEL: {best_model_name} ===")
print(f"Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
print(f"Test F1-Score: {results[best_model_name]['test_f1']:.4f}")

# Ensemble method
print("\n=== ENSEMBLE METHOD ===")
# Select top 3 models based on CV accuracy
top_models = sorted(results.items(), key=lambda x: x[1]['cv_accuracy'], reverse=True)[:3]
ensemble_models = [(name, result['model']) for name, result in top_models]

ensemble = VotingClassifier(
    estimators=ensemble_models,
    voting='soft'
)
ensemble.fit(X_train_scaled, y_train_balanced)
ensemble_pred = ensemble.predict(X_test_scaled)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')

print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
print(f"Ensemble F1-Score: {ensemble_f1:.4f}")

# Use ensemble if it's better, otherwise use best individual model
if ensemble_accuracy > results[best_model_name]['test_accuracy']:
    final_model = ensemble
    final_predictions = ensemble_pred
    final_accuracy = ensemble_accuracy
    final_model_name = "Ensemble"
else:
    final_model = best_model
    final_predictions = best_predictions
    final_accuracy = results[best_model_name]['test_accuracy']
    final_model_name = best_model_name

print(f"\n=== FINAL MODEL: {final_model_name} ===")
print(f"Final Accuracy: {final_accuracy:.4f}")

# Detailed evaluation
print("\n=== DETAILED EVALUATION ===")
reverse_cancer_mapping = {v: k for k, v in cancer_mapping.items()}
unique_classes = np.unique(np.concatenate([y_test, final_predictions]))
target_names = [reverse_cancer_mapping[i] for i in sorted(unique_classes)]

print("Confusion Matrix:")
cm = confusion_matrix(y_test, final_predictions)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, final_predictions, 
                          labels=unique_classes, target_names=target_names))

# Feature importance (for tree-based models)
if hasattr(final_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': final_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(feature_importance.head(10))

# Visualizations
plt.figure(figsize=(15, 10))

# 1. Model comparison
plt.subplot(2, 3, 1)
model_names = list(results.keys())
accuracies = [results[name]['test_accuracy'] for name in model_names]
plt.barh(model_names, accuracies)
plt.title('Model Comparison (Test Accuracy)')
plt.xlabel('Accuracy')

# 2. Confusion matrix
plt.subplot(2, 3, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 3. Feature importance
if hasattr(final_model, 'feature_importances_'):
    plt.subplot(2, 3, 3)
    top_features = feature_importance.head(10)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')

# 4. Class distribution
plt.subplot(2, 3, 4)
original_dist = df['Cancer_Type(AML, ALL, CLL)'].value_counts()
cancer_names = [reverse_cancer_mapping[i] for i in original_dist.index]
plt.pie(original_dist.values, labels=cancer_names, autopct='%1.1f%%')
plt.title('Cancer Type Distribution')

# 5. CV scores comparison
plt.subplot(2, 3, 5)
cv_means = [results[name]['cv_accuracy'] for name in model_names]
cv_stds = [results[name]['cv_accuracy_std'] for name in model_names]
plt.errorbar(range(len(model_names)), cv_means, yerr=cv_stds, fmt='o')
plt.xticks(range(len(model_names)), model_names, rotation=45)
plt.title('Cross-Validation Accuracy')
plt.ylabel('CV Accuracy')

# 6. Results summary
plt.subplot(2, 3, 6)
metrics = ['CV Accuracy', 'Test Accuracy', 'Test F1-Score']
best_values = [
    results[best_model_name]['cv_accuracy'],
    results[best_model_name]['test_accuracy'],
    results[best_model_name]['test_f1']
]
plt.bar(metrics, best_values)
plt.title(f'Best Model Performance: {best_model_name}')
plt.ylabel('Score')

plt.tight_layout()
plt.savefig('public/comprehensive_analysis/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Save results
print("\n=== SAVING RESULTS ===")
joblib.dump(final_model, 'public/comprehensive_analysis/best_cancer_model.pkl')
joblib.dump(scaler, 'public/comprehensive_analysis/scaler.pkl')
joblib.dump(smote, 'public/comprehensive_analysis/smote.pkl')

# Save comprehensive results
comprehensive_results = {
    'final_model': final_model_name,
    'final_accuracy': final_accuracy,
    'all_model_results': {name: {
        'cv_accuracy': results[name]['cv_accuracy'],
        'test_accuracy': results[name]['test_accuracy'],
        'test_f1': results[name]['test_f1']
    } for name in results.keys()},
    'selected_features': selected_features.tolist(),
    'cancer_mapping': cancer_mapping,
    'class_distribution': df['Cancer_Type(AML, ALL, CLL)'].value_counts().to_dict(),
    'feature_importance': feature_importance.to_dict('records') if hasattr(final_model, 'feature_importances_') else None
}

import json
with open('public/comprehensive_analysis/comprehensive_results.json', 'w') as f:
    json.dump(comprehensive_results, f, indent=2)

print(f"Comprehensive analysis complete!")
print(f"Final model: {final_model_name}")
print(f"Final accuracy: {final_accuracy:.4f}")
print(f"Improvement over original: {(final_accuracy - 0.18):.4f}")