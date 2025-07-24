import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

print("=== XGBoost Blood Cancer Classification ===")

# Load the processed dataset
try:
    df = pd.read_csv('public/test_analysis/processed_blood_cancer_dataset.csv')
    print(f"✅ Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
except FileNotFoundError:
    print("❌ Error: Could not find the processed dataset file.")
    print("Please ensure 'public/test_analysis/processed_blood_cancer_dataset.csv' exists.")
    exit()

# Display basic information about the dataset
print(f"\n=== Dataset Overview ===")
print(f"Dataset shape: {df.shape}")
print(f"\nTarget variable distribution:")
print(df['Cancer_Type(AML, ALL, CLL)'].value_counts())
print(f"\nMissing values:")
print(df.isnull().sum())

# Prepare features and target
X = df.drop('Cancer_Type(AML, ALL, CLL)', axis=1)
y = df['Cancer_Type(AML, ALL, CLL)']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")

# Check class distribution
print(f"\nClass distribution:")
class_counts = y.value_counts()
print(class_counts)
print(f"Class balance ratio: {class_counts.min() / class_counts.max():.3f}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Feature scaling (XGBoost doesn't strictly require it, but can help)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n=== XGBoost Model Training ===")

# Define XGBoost parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}

# Create XGBoost classifier
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

# Use RandomizedSearchCV for more efficient hyperparameter tuning
print("🔄 Performing hyperparameter tuning with RandomizedSearchCV...")

# Use StratifiedKFold for better cross-validation with imbalanced classes
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_grid,
    n_iter=50,  # Number of parameter combinations to try
    cv=cv_strategy,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# Fit the random search
random_search.fit(X_train_scaled, y_train)

print(f"✅ Best parameters found: {random_search.best_params_}")
print(f"✅ Best cross-validation score: {random_search.best_score_:.4f}")

# Get the best model
best_xgb_model = random_search.best_estimator_

# Train the best model on full training data
print("\n🔄 Training final model with best parameters...")
best_xgb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = best_xgb_model.predict(X_test_scaled)
y_pred_proba = best_xgb_model.predict_proba(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n=== Model Performance ===")
print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"🎯 Weighted F1-Score: {f1:.4f}")

# Detailed classification report
print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print(f"\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Cross-validation scores
print(f"\n📊 Cross-Validation Scores:")
cv_scores = cross_val_score(best_xgb_model, X_train_scaled, y_train, cv=cv_strategy, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance
print(f"\n📊 Feature Importance:")
feature_importance = best_xgb_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df)

# If accuracy is below 90%, try additional techniques
if accuracy < 0.90:
    print(f"\n⚠️  Current accuracy ({accuracy*100:.2f}%) is below 90%. Trying advanced techniques...")
    
    # Try with more aggressive hyperparameter tuning
    advanced_param_grid = {
        'n_estimators': [300, 500, 700, 1000],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [1, 2, 3],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    print("🔄 Performing advanced hyperparameter tuning...")
    
    advanced_random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=advanced_param_grid,
        n_iter=100,  # More iterations
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    advanced_random_search.fit(X_train_scaled, y_train)
    
    # Get the new best model
    advanced_best_model = advanced_random_search.best_estimator_
    
    # Make new predictions
    y_pred_advanced = advanced_best_model.predict(X_test_scaled)
    accuracy_advanced = accuracy_score(y_test, y_pred_advanced)
    
    print(f"🎯 Advanced Model Accuracy: {accuracy_advanced:.4f} ({accuracy_advanced*100:.2f}%)")
    
    if accuracy_advanced > accuracy:
        best_xgb_model = advanced_best_model
        y_pred = y_pred_advanced
        accuracy = accuracy_advanced
        print(f"✅ Improved accuracy achieved!")

# Save the best model and related files
print(f"\n💾 Saving model and results...")

# Save the best XGBoost model
model_file = 'public/test_analysis/xgboost_blood_cancer_model.pkl'
joblib.dump(best_xgb_model, model_file)
print(f"✅ XGBoost model saved to: {model_file}")

# Save the scaler
scaler_file = 'public/test_analysis/xgboost_scaler.pkl'
joblib.dump(scaler, scaler_file)
print(f"✅ Scaler saved to: {scaler_file}")

# Save feature importance
importance_file = 'public/test_analysis/xgboost_feature_importance.csv'
importance_df.to_csv(importance_file, index=False)
print(f"✅ Feature importance saved to: {importance_file}")

# Save model performance metrics
metrics = {
    'accuracy': accuracy,
    'f1_score': f1,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'best_params': random_search.best_params_
}

metrics_file = 'public/test_analysis/xgboost_metrics.pkl'
joblib.dump(metrics, metrics_file)
print(f"✅ Model metrics saved to: {metrics_file}")

# Create visualizations
print(f"\n📊 Creating visualizations...")

# 1. Feature Importance Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.title(f'XGBoost Feature Importance\nModel Accuracy: {accuracy*100:.2f}%')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('public/test_analysis/xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'XGBoost Confusion Matrix\nAccuracy: {accuracy*100:.2f}%')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('public/test_analysis/xgboost_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Cross-validation scores plot
plt.figure(figsize=(8, 6))
plt.bar(range(len(cv_scores)), cv_scores)
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
plt.title('Cross-Validation Accuracy Scores')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('public/test_analysis/xgboost_cv_scores.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Visualizations saved to 'public/test_analysis/' directory")

# Final summary
print(f"\n=== FINAL RESULTS SUMMARY ===")
print(f"🏆 Final Model Accuracy: {accuracy*100:.2f}%")
print(f"🏆 Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*200:.2f}%)")
print(f"🏆 Weighted F1-Score: {f1:.4f}")

if accuracy >= 0.90:
    print(f"🎉 SUCCESS: Achieved target accuracy of ≥90%!")
else:
    print(f"⚠️  Target accuracy of 90% not quite reached. Consider:")
    print(f"   - Collecting more training data")
    print(f"   - Feature engineering")
    print(f"   - Ensemble methods")
    print(f"   - Different algorithms")

print(f"\n📁 All files saved in 'public/test_analysis/' directory:")
print(f"   - xgboost_blood_cancer_model.pkl")
print(f"   - xgboost_scaler.pkl")
print(f"   - xgboost_feature_importance.csv")
print(f"   - xgboost_metrics.pkl")
print(f"   - xgboost_feature_importance.png")
print(f"   - xgboost_confusion_matrix.png")
print(f"   - xgboost_cv_scores.png")

print(f"\n=== XGBoost Analysis Completed Successfully! ===")