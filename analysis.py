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

# Fill missing values for categorical variables
df['Genetic_Data(BCR-ABL, FLT3)'] = df['Genetic_Data(BCR-ABL, FLT3)'].fillna('Unknown')
df['Side_Effects'] = df['Side_Effects'].fillna('Unknown')

# For any remaining missing values in numerical columns
numerical_columns = df.select_dtypes(include=[np.number]).columns
for col in numerical_columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

print("\nMissing values after preprocessing:")
print(df.isnull().sum())

# 2. Encode Categorical Variables
print("\n=== ENCODING CATEGORICAL VARIABLES ===")

# Create label encoders for categorical variables
label_encoders = {}
categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    if col not in ['Cancer_Type(AML, ALL, CLL)', 'Treatment_Outcome']:  # Don't encode target variables yet
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"Encoded {col}: {le.classes_}")

# 3. Prepare features and targets
print("\n=== FEATURE PREPARATION ===")

# Select features (excluding original categorical columns and target variables)
feature_columns = [col for col in df.columns if col.endswith('_encoded') or col in numerical_columns]
target_columns = ['Cancer_Type(AML, ALL, CLL)', 'Treatment_Outcome']

# Remove target columns from features if they exist
feature_columns = [col for col in feature_columns if col not in target_columns]

print(f"Selected features: {feature_columns}")

X = df[feature_columns]
print(f"Feature matrix shape: {X.shape}")

# 4. Create targets for different prediction tasks
print("\n=== TARGET PREPARATION ===")

# Target 1: Cancer Type Prediction
le_cancer = LabelEncoder()
y_cancer = le_cancer.fit_transform(df['Cancer_Type(AML, ALL, CLL)'])
print(f"Cancer types: {le_cancer.classes_}")
print(f"Cancer type distribution: {pd.Series(y_cancer).value_counts()}")

# Target 2: Treatment Outcome Prediction
le_outcome = LabelEncoder()
y_outcome = le_outcome.fit_transform(df['Treatment_Outcome'])
print(f"Treatment outcomes: {le_outcome.classes_}")
print(f"Treatment outcome distribution: {pd.Series(y_outcome).value_counts()}")

# 5. Feature Scaling
print("\n=== FEATURE SCALING ===")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Scaled feature matrix shape: {X_scaled.shape}")

# 6. Feature Selection
print("\n=== FEATURE SELECTION ===")

def perform_feature_selection(X, y, feature_names, task_name):
    print(f"\n--- Feature Selection for {task_name} ---")
    
    # Method 1: SelectKBest with f_classif
    k_best = SelectKBest(score_func=f_classif, k=min(10, X.shape[1]))
    X_selected = k_best.fit_transform(X, y)
    selected_features = [feature_names[i] for i in k_best.get_support(indices=True)]
    scores = k_best.scores_[k_best.get_support()]
    
    print(f"Top features selected by SelectKBest:")
    for feature, score in zip(selected_features, scores):
        print(f"  {feature}: {score:.2f}")
    
    # Method 2: Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 features by Random Forest importance:")
    print(feature_importance.head(10))
    
    return X_selected, selected_features, feature_importance

# Perform feature selection for both tasks
X_cancer_selected, features_cancer, importance_cancer = perform_feature_selection(
    X_scaled, y_cancer, feature_columns, "Cancer Type Prediction"
)

X_outcome_selected, features_outcome, importance_outcome = perform_feature_selection(
    X_scaled, y_outcome, feature_columns, "Treatment Outcome Prediction"
)

# 7. SVM Model Training and Evaluation
print("\n=== SVM MODEL TRAINING ===")

def train_evaluate_svm(X, y, target_name, class_names):
    print(f"\n--- SVM Training for {target_name} ---")
    
    # Check class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
    
    # Filter out classes with very few samples (less than 2)
    min_samples_per_class = 2
    valid_classes = unique_classes[class_counts >= min_samples_per_class]
    
    if len(valid_classes) < len(unique_classes):
        print(f"Filtering out classes with < {min_samples_per_class} samples")
        valid_mask = np.isin(y, valid_classes)
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Update class names
        class_names = [class_names[i] for i in valid_classes]
        
        # Re-encode labels to be consecutive
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_classes)}
        y = np.array([label_mapping[label] for label in y])
        
        print(f"Filtered dataset shape: {X.shape}")
        print(f"Updated class distribution: {dict(zip(range(len(valid_classes)), np.bincount(y)))}")
    
    # Split the data - use stratify only if all classes have at least 2 samples
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # If stratify still fails, do random split
        print("Stratified split failed, using random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Check if we have enough samples for grid search
    if X_train.shape[0] < 50:
        print("Small dataset detected, using simplified parameter grid")
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
        cv_folds = min(3, len(np.unique(y_train)))
    else:
        # Full parameter grid for larger datasets
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly']
        }
        cv_folds = 5
    
    print(f"Using {cv_folds}-fold cross-validation")
    print("Performing grid search for optimal hyperparameters...")
    
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(
        svm, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Train final model with best parameters
    best_svm = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_svm.predict(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Cross-validation scores (only if we have enough samples)
    if X_train.shape[0] >= 10 and len(np.unique(y_train)) >= 2:
        cv_scores = cross_val_score(best_svm, X_train, y_train, cv=cv_folds)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    else:
        print("Skipping cross-validation due to small dataset size")
    
    # Classification report
    print(f"\nClassification Report for {target_name}:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix for {target_name}:")
    print(cm)
    
    return best_svm, accuracy, y_test, y_pred

# Train SVM for Cancer Type Prediction
print("\n" + "="*50)
print("CANCER TYPE PREDICTION")
print("="*50)

svm_cancer, acc_cancer, y_test_cancer, y_pred_cancer = train_evaluate_svm(
    X_cancer_selected, y_cancer, 
    "Cancer Type Prediction", 
    le_cancer.classes_
)

# Train SVM for Treatment Outcome Prediction
print("\n" + "="*50)
print("TREATMENT OUTCOME PREDICTION")
print("="*50)

svm_outcome, acc_outcome, y_test_outcome, y_pred_outcome = train_evaluate_svm(
    X_outcome_selected, y_outcome, 
    "Treatment Outcome Prediction", 
    le_outcome.classes_
)

# 8. Model Comparison and Results Summary
print("\n" + "="*50)
print("MODEL PERFORMANCE SUMMARY")
print("="*50)

print(f"Cancer Type Prediction Accuracy: {acc_cancer:.4f}")
print(f"Treatment Outcome Prediction Accuracy: {acc_outcome:.4f}")

# 9. Save models and preprocessors
print("\n=== SAVING MODELS ===")

# Save models
joblib.dump(svm_cancer, 'svm_cancer_model.pkl')
joblib.dump(svm_outcome, 'svm_outcome_model.pkl')

# Save preprocessors
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_cancer, 'label_encoder_cancer.pkl')
joblib.dump(le_outcome, 'label_encoder_outcome.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("Models and preprocessors saved successfully!")

# 10. Feature Importance Analysis
print("\n=== FEATURE IMPORTANCE ANALYSIS ===")

print("\nTop 10 Most Important Features for Cancer Type Prediction:")
print(importance_cancer.head(10))

print("\nTop 10 Most Important Features for Treatment Outcome Prediction:")
print(importance_outcome.head(10))

# 11. Prediction Function
def predict_new_sample(sample_data, model_type='cancer'):
    """
    Function to make predictions on new samples
    sample_data: dictionary with feature values
    model_type: 'cancer' or 'outcome'
    """
    if model_type == 'cancer':
        model = svm_cancer
        encoder = le_cancer
        features = features_cancer
    else:
        model = svm_outcome
        encoder = le_outcome
        features = features_outcome
    
    # Process sample data (this would need to be adapted based on your specific features)
    # For now, this is a placeholder
    print(f"Prediction function ready for {model_type} prediction")
    
    return None

print("\n=== ANALYSIS COMPLETE ===")
print("Key Insights:")
print("1. Data preprocessing completed with missing value handling")
print("2. Categorical variables encoded successfully")
print("3. Feature selection performed using multiple methods")
print("4. SVM models trained with hyperparameter tuning")
print("5. Models saved for future use")
print("\nNext steps:")
print("1. Analyze feature importance for clinical insights")
print("2. Validate models on additional test data")
print("3. Deploy models for clinical decision support")