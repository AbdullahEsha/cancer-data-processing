import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

print("=== SVM Blood Cancer Classification (4 Features Only) ===")

# Load the dataset with only required columns
try:
    # Define the columns we want to keep
    required_columns = [
        'Platelet Count( (/cumm)',
        'Total WBC count(/cumm)', 
        'Age',
        'Cancer_Type(AML, ALL, CLL)'
    ]
    
    # Load only the required columns
    df = pd.read_csv('public/blood_cancer_diseases_dataset.csv', usecols=required_columns)
    print(f"✅ Dataset loaded successfully with selected columns!")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
except FileNotFoundError:
    print("❌ Error: Could not find the dataset file.")
    print("Please ensure 'blood_cancer_diseases_dataset.csv' exists in the current directory.")
    exit()
except KeyError as e:
    print(f"❌ Error: Could not find required columns. Available columns might be named differently.")
    print(f"Missing column: {e}")
    # Try to load the full dataset to see available columns
    try:
        df_full = pd.read_csv('public/blood_cancer_diseases_dataset.csv')
        print("Available columns in the dataset:")
        for i, col in enumerate(df_full.columns):
            print(f"{i+1}. {col}")
    except:
        pass
    exit()

# Clean column names for easier handling
def clean_column_names(df):
    """Clean column names for easier handling"""
    df_clean = df.copy()
    
    # Create a mapping for column names
    column_mapping = {
        'Platelet Count( (/cumm)': 'Platelet_Count',
        'Total WBC count(/cumm)': 'WBC_Count',
        'Age': 'Age',
        'Cancer_Type(AML, ALL, CLL)': 'Cancer_Type'
    }
    
    # Rename columns
    df_clean = df_clean.rename(columns=column_mapping)
    
    return df_clean, column_mapping

# Clean the column names
df, column_mapping = clean_column_names(df)
print(f"\nCleaned columns: {df.columns.tolist()}")

# Display basic information about the dataset
print(f"\n=== Dataset Overview ===")
print(f"Dataset shape: {df.shape}")
print(f"Data types:\n{df.dtypes}")

# Check for missing values
print(f"\nMissing values:")
missing_values = df.isnull().sum()
print(missing_values)

# Handle missing values
print(f"\n=== Data Preprocessing ===")
if missing_values.sum() > 0:
    print("Handling missing values...")
    # Fill numerical columns with median
    for col in ['Platelet_Count', 'WBC_Count', 'Age']:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical columns with mode
    if df['Cancer_Type'].isnull().sum() > 0:
        df['Cancer_Type'] = df['Cancer_Type'].fillna(df['Cancer_Type'].mode()[0])
    
    print("Missing values handled!")

# Display target variable distribution
print(f"\nTarget variable (Cancer_Type) distribution:")
target_counts = df['Cancer_Type'].value_counts()
print(target_counts)
print(f"Number of classes: {len(target_counts)}")

# Basic statistics
print(f"\nBasic statistics:")
print(df.describe())

# Data Distribution Visualization
def plot_data_distribution(df):
    """Plot comprehensive data distribution for 4 features"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Blood Cancer Data Distribution Analysis (4 Features)', fontsize=16)
    
    # Age distribution
    axes[0,0].hist(df['Age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Age Distribution')
    axes[0,0].set_xlabel('Age')
    axes[0,0].set_ylabel('Frequency')
    
    # WBC count distribution
    axes[0,1].hist(df['WBC_Count'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0,1].set_title('WBC Count Distribution')
    axes[0,1].set_xlabel('WBC Count')
    axes[0,1].set_ylabel('Frequency')
    
    # Platelet count distribution
    axes[0,2].hist(df['Platelet_Count'], bins=20, alpha=0.7, color='salmon', edgecolor='black')
    axes[0,2].set_title('Platelet Count Distribution')
    axes[0,2].set_xlabel('Platelet Count')
    axes[0,2].set_ylabel('Frequency')
    
    # Cancer type distribution
    df['Cancer_Type'].value_counts().plot(kind='bar', ax=axes[1,0], color='orange', alpha=0.7)
    axes[1,0].set_title('Cancer Type Distribution')
    axes[1,0].set_xlabel('Cancer Type')
    axes[1,0].set_ylabel('Count')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Correlation heatmap
    correlation_matrix = df[['Age', 'WBC_Count', 'Platelet_Count']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
    axes[1,1].set_title('Feature Correlation Matrix')
    
    # Box plots for outlier detection
    df[['Age', 'WBC_Count', 'Platelet_Count']].boxplot(ax=axes[1,2])
    axes[1,2].set_title('Box Plots for Outlier Detection')
    axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('data_distribution_4features.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot data distribution
print(f"\n=== Data Visualization ===")
plot_data_distribution(df)

# Outlier Detection and Removal using 3 numerical features
def detect_and_remove_outliers(df, contamination=0.05):
    """Detect and remove outliers using Isolation Forest on numerical features"""
    
    # Select numerical features only
    numerical_features = ['Age', 'WBC_Count', 'Platelet_Count']
    X_numerical = df[numerical_features]
    
    print(f"Applying outlier detection on features: {numerical_features}")
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_labels = iso_forest.fit_predict(X_numerical)
    
    # Identify outliers
    outlier_mask = outlier_labels == -1
    outlier_indices = np.where(outlier_mask)[0]
    
    print(f"Detected {len(outlier_indices)} outliers ({len(outlier_indices)/len(df)*100:.2f}% of data)")
    
    # Visualize outliers
    plt.figure(figsize=(15, 5))
    
    # 3D scatter plot
    from mpl_toolkits.mplot3d import Axes3D
    
    # 2D projections
    plt.subplot(1, 3, 1)
    normal_mask = ~outlier_mask
    plt.scatter(df.loc[normal_mask, 'Age'], df.loc[normal_mask, 'WBC_Count'], 
               alpha=0.6, label='Normal', color='blue', s=30)
    plt.scatter(df.loc[outlier_mask, 'Age'], df.loc[outlier_mask, 'WBC_Count'], 
               alpha=0.8, label='Outliers', color='red', marker='x', s=100)
    plt.xlabel('Age')
    plt.ylabel('WBC Count')
    plt.title('Age vs WBC Count')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.scatter(df.loc[normal_mask, 'Age'], df.loc[normal_mask, 'Platelet_Count'], 
               alpha=0.6, label='Normal', color='blue', s=30)
    plt.scatter(df.loc[outlier_mask, 'Age'], df.loc[outlier_mask, 'Platelet_Count'], 
               alpha=0.8, label='Outliers', color='red', marker='x', s=100)
    plt.xlabel('Age')
    plt.ylabel('Platelet Count')
    plt.title('Age vs Platelet Count')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.scatter(df.loc[normal_mask, 'WBC_Count'], df.loc[normal_mask, 'Platelet_Count'], 
               alpha=0.6, label='Normal', color='blue', s=30)
    plt.scatter(df.loc[outlier_mask, 'WBC_Count'], df.loc[outlier_mask, 'Platelet_Count'], 
               alpha=0.8, label='Outliers', color='red', marker='x', s=100)
    plt.xlabel('WBC Count')
    plt.ylabel('Platelet Count')
    plt.title('WBC vs Platelet Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('outlier_detection_4features.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Remove outliers
    df_clean = df[~outlier_mask].copy()
    
    return df_clean, outlier_indices

# Apply outlier detection and removal
print(f"\n=== Outlier Detection and Removal ===")
df_clean, outlier_indices = detect_and_remove_outliers(df, contamination=0.05)
print(f"Dataset shape after outlier removal: {df_clean.shape}")

# Encode the target variable
label_encoder = LabelEncoder()
df_clean['Cancer_Type_Encoded'] = label_encoder.fit_transform(df_clean['Cancer_Type'])

print(f"\nTarget encoding:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name} -> {i}")

# Prepare features and target
X = df_clean[['Age', 'WBC_Count', 'Platelet_Count']]
y = df_clean['Cancer_Type_Encoded']

print(f"\nFinal dataset info:")
print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Features: {X.columns.tolist()}")

# Check class distribution after outlier removal
print(f"\nClass distribution after cleaning:")
class_counts = y.value_counts().sort_index()
for i, count in enumerate(class_counts):
    print(f"{label_encoder.classes_[i]}: {count} samples")

class_balance = class_counts.min() / class_counts.max()
print(f"Class balance ratio: {class_balance:.3f}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nData split:")
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Feature scaling (crucial for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Features scaled using StandardScaler")

print(f"\n=== SVM Model Training ===")

# Define SVM parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'degree': [2, 3, 4],  # for polynomial kernel
    'class_weight': [None, 'balanced']  # to handle class imbalance
}

# Create SVM classifier
svm_model = SVC(
    random_state=42,
    probability=True  # Enable probability estimates
)

# Use GridSearchCV for thorough hyperparameter tuning (smaller search space)
print("🔄 Performing hyperparameter tuning with GridSearchCV...")

# Use StratifiedKFold for better cross-validation
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# For smaller dataset, we can use GridSearchCV
grid_search = GridSearchCV(
    svm_model,
    param_grid,
    cv=cv_strategy,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit the grid search
grid_search.fit(X_train_scaled, y_train)

print(f"✅ Best parameters found: {grid_search.best_params_}")
print(f"✅ Best cross-validation score: {grid_search.best_score_:.4f}")

# Get the best model
best_svm_model = grid_search.best_estimator_

# Make predictions
y_pred = best_svm_model.predict(X_test_scaled)
y_pred_proba = best_svm_model.predict_proba(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n=== Model Performance ===")
print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"🎯 Weighted F1-Score: {f1:.4f}")

# Check for overfitting
train_pred = best_svm_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, train_pred)
print(f"🎯 Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"🎯 Overfitting Check: {abs(train_accuracy - accuracy):.4f}")

if abs(train_accuracy - accuracy) < 0.05:
    print("✅ Model shows good generalization (low overfitting)")
elif abs(train_accuracy - accuracy) < 0.10:
    print("⚠️ Model shows slight overfitting")
else:
    print("❌ Model shows significant overfitting")

# Detailed classification report
print(f"\n📊 Detailed Classification Report:")
target_names = label_encoder.classes_
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
print(f"\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Cross-validation scores
print(f"\n📊 Cross-Validation Analysis:")
cv_scores = cross_val_score(best_svm_model, X_train_scaled, y_train, cv=cv_strategy, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"Individual CV scores: {cv_scores}")

# Feature importance (for linear kernel only)
if best_svm_model.kernel == 'linear':
    print(f"\n📊 Feature Importance (Linear SVM):")
    feature_importance = abs(best_svm_model.coef_[0])  # Take first class coefficients
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    print(importance_df)
else:
    print(f"\n📊 Feature importance not available for {best_svm_model.kernel} kernel")
    importance_df = pd.DataFrame({'feature': X.columns, 'importance': [0]*len(X.columns)})

# Create output directory
import os
os.makedirs('public/svm_results', exist_ok=True)

# Save the model and results
print(f"\n💾 Saving model and results...")

# Save the best SVM model
model_file = 'public/svm_results/svm_blood_cancer_model_4features.pkl'
joblib.dump(best_svm_model, model_file)
print(f"✅ SVM model saved to: {model_file}")

# Save the scaler
scaler_file = 'public/svm_results/svm_scaler_4features.pkl'
joblib.dump(scaler, scaler_file)
print(f"✅ Scaler saved to: {scaler_file}")

# Save the label encoder
encoder_file = 'public/svm_results/label_encoder_4features.pkl'
joblib.dump(label_encoder, encoder_file)
print(f"✅ Label encoder saved to: {encoder_file}")

# Save model performance metrics
metrics = {
    'accuracy': accuracy,
    'f1_score': f1,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'best_params': grid_search.best_params_,
    'train_accuracy': train_accuracy,
    'overfitting_measure': abs(train_accuracy - accuracy),
    'n_outliers_removed': len(outlier_indices),
    'final_dataset_size': len(df_clean),
    'feature_names': X.columns.tolist(),
    'class_names': label_encoder.classes_.tolist()
}

metrics_file = 'public/svm_results/svm_metrics_4features.pkl'
joblib.dump(metrics, metrics_file)
print(f"✅ Model metrics saved to: {metrics_file}")

# Create visualizations
print(f"\n📊 Creating visualizations...")

# 1. Feature Importance Plot (if linear kernel)
if best_svm_model.kernel == 'linear':
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.title(f'SVM Feature Importance (Linear Kernel)\nAccuracy: {accuracy*100:.2f}%')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('public/svm_results/svm_feature_importance_4features.png', dpi=300, bbox_inches='tight')
    plt.show()

# 2. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, 
            yticklabels=target_names)
plt.title(f'SVM Confusion Matrix\nAccuracy: {accuracy*100:.2f}%')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('public/svm_results/svm_confusion_matrix_4features.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Cross-validation scores plot
plt.figure(figsize=(8, 6))
plt.bar(range(len(cv_scores)), cv_scores, alpha=0.7, color='skyblue')
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', 
           label=f'Mean: {cv_scores.mean():.4f}')
plt.axhline(y=cv_scores.mean() + cv_scores.std(), color='orange', linestyle=':', alpha=0.7)
plt.axhline(y=cv_scores.mean() - cv_scores.std(), color='orange', linestyle=':', alpha=0.7)
plt.title('SVM Cross-Validation Accuracy Scores')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('public/svm_results/svm_cv_scores_4features.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Decision boundary visualization (3D with 3 features)
print("🔄 Creating 3D decision boundary visualization...")

# Create a simplified visualization using first 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)
X_pca_test = pca.transform(X_test_scaled)

# Train a simple SVM on PCA components for visualization
svm_viz = SVC(kernel=best_svm_model.kernel, C=best_svm_model.C, gamma=best_svm_model.gamma)
svm_viz.fit(X_pca, y_train)

# Create a mesh
h = 0.02
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Make predictions on the mesh
Z = svm_viz.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set3)
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i, class_name in enumerate(label_encoder.classes_):
    mask = y_train == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=colors[i % len(colors)], label=class_name, 
               edgecolors='black', alpha=0.7)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title(f'SVM Decision Boundary (PCA Space)\nKernel: {best_svm_model.kernel}, Accuracy: {accuracy*100:.2f}%')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('public/svm_results/svm_decision_boundary_4features.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ All visualizations saved to 'public/svm_results/' directory")

# Final comprehensive summary
print(f"\n" + "="*60)
print(f"FINAL SVM RESULTS SUMMARY (4 Features)")
print(f"="*60)
print(f"🏆 Algorithm: Support Vector Machine (SVM)")
print(f"🏆 Dataset: 4 features only (Age, WBC_Count, Platelet_Count, Cancer_Type)")
print(f"🏆 Original dataset size: {len(df)} samples")
print(f"🏆 After outlier removal: {len(df_clean)} samples")
print(f"🏆 Outliers removed: {len(outlier_indices)} ({len(outlier_indices)/len(df)*100:.2f}%)")
print(f"🏆 Number of classes: {len(label_encoder.classes_)}")
print(f"🏆 Class names: {', '.join(label_encoder.classes_)}")
print(f"🏆 Best kernel: {best_svm_model.kernel}")
print(f"🏆 Best C parameter: {best_svm_model.C}")
print(f"🏆 Best gamma: {best_svm_model.gamma}")
print(f"🏆 Final Test Accuracy: {accuracy*100:.2f}%")
print(f"🏆 Training Accuracy: {train_accuracy*100:.2f}%")
print(f"🏆 Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*200:.2f}%)")
print(f"🏆 Weighted F1-Score: {f1:.4f}")
print(f"🏆 Overfitting Measure: {abs(train_accuracy - accuracy):.4f}")

# Performance assessment
if accuracy >= 0.90:
    print(f"🎉 EXCELLENT: Achieved >90% accuracy!")
elif accuracy >= 0.80:
    print(f"✅ GOOD: Achieved >80% accuracy!")
elif accuracy >= 0.70:
    print(f"👍 FAIR: Achieved >70% accuracy!")
else:
    print(f"⚠️ NEEDS IMPROVEMENT: Consider more data or feature engineering")

if abs(train_accuracy - accuracy) < 0.05:
    print(f"✅ LOW OVERFITTING: Model generalizes well!")
elif abs(train_accuracy - accuracy) < 0.10:
    print(f"⚠️ MODERATE OVERFITTING: Acceptable for small dataset")
else:
    print(f"❌ HIGH OVERFITTING: Consider regularization")

print(f"\n📁 All files saved in 'public/svm_results/' directory:")
print(f"   - svm_blood_cancer_model_4features.pkl")
print(f"   - svm_scaler_4features.pkl") 
print(f"   - label_encoder_4features.pkl")
print(f"   - svm_metrics_4features.pkl")
print(f"   - Various visualization PNG files")

print(f"\n=== SVM Analysis Completed Successfully! ===")

# Quick model testing function
def test_prediction(age, wbc_count, platelet_count):
    """Test the model with new data"""
    print(f"\n=== Testing Model Prediction ===")
    
    # Create test sample
    test_sample = np.array([[age, wbc_count, platelet_count]])
    
    # Scale the features
    test_sample_scaled = scaler.transform(test_sample)
    
    # Make prediction
    prediction = best_svm_model.predict(test_sample_scaled)[0]
    probabilities = best_svm_model.predict_proba(test_sample_scaled)[0]
    
    # Convert prediction back to class name
    predicted_class = label_encoder.inverse_transform([prediction])[0]
    
    print(f"Input: Age={age}, WBC={wbc_count}, Platelet={platelet_count}")
    print(f"Predicted Cancer Type: {predicted_class}")
    print(f"Prediction probabilities:")
    for i, prob in enumerate(probabilities):
        class_name = label_encoder.classes_[i]
        print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")

# Example prediction (you can modify these values)
if len(df_clean) > 0:
    # Use median values from the dataset as an example
    example_age = df_clean['Age'].median()
    example_wbc = df_clean['WBC_Count'].median()
    example_platelet = df_clean['Platelet_Count'].median()
    
    test_prediction(example_age, example_wbc, example_platelet)