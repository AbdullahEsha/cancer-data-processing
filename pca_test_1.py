import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# XGBoost and advanced ML libraries
import xgboost as xgb
from sklearn.model_selection import (train_test_split, GridSearchCV, StratifiedKFold, 
                                   cross_val_score, validation_curve, learning_curve)
from sklearn.preprocessing import (LabelEncoder, StandardScaler, MinMaxScaler, 
                                 RobustScaler, PolynomialFeatures, PowerTransformer)
from sklearn.feature_selection import (SelectKBest, f_classif, chi2, mutual_info_classif,
                                     RFE, SelectFromModel, VarianceThreshold)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Metrics and evaluation
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                           roc_auc_score, precision_recall_fscore_support, roc_curve, 
                           auc, f1_score, precision_score, recall_score, log_loss)

# Additional libraries
from scipy import stats
from scipy.stats import boxcox, yeojohnson
import time
import joblib

class AdvancedFeatureEngineering:
    """Advanced feature engineering class for blood cancer prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def create_interaction_features(self, df):
        """Create interaction features between key variables"""
        df_engineered = df.copy()
        
        # Ratio features with safe division
        df_engineered['WBC_Platelet_Ratio'] = df_engineered['Total WBC count(/cumm)'] / (df_engineered['Platelet Count(/cumm)'] + 1)
        df_engineered['Age_WBC_Interaction'] = df_engineered['Age'] * np.log1p(df_engineered['Total WBC count(/cumm)'])
        df_engineered['Age_Platelet_Interaction'] = df_engineered['Age'] * np.log1p(df_engineered['Platelet Count(/cumm)'])
        
        # Log transformations for skewed data
        df_engineered['Log_WBC'] = np.log1p(df_engineered['Total WBC count(/cumm)'])
        df_engineered['Log_Platelet'] = np.log1p(df_engineered['Platelet Count(/cumm)'])
        df_engineered['Sqrt_Age'] = np.sqrt(df_engineered['Age'])
        
        # Binning age into categories
        df_engineered['Age_Group'] = pd.cut(df_engineered['Age'], bins=[0, 40, 60, 80, 100], labels=[0, 1, 2, 3])
        df_engineered['Age_Group'] = df_engineered['Age_Group'].astype(int)
        
        # WBC count categories (based on medical ranges)
        df_engineered['WBC_Category'] = pd.cut(df_engineered['Total WBC count(/cumm)'], 
                                              bins=[0, 4000, 11000, 50000, np.inf], 
                                              labels=[0, 1, 2, 3])  # Low, Normal, High, Very High
        df_engineered['WBC_Category'] = df_engineered['WBC_Category'].astype(int)
        
        # Platelet count categories
        df_engineered['Platelet_Category'] = pd.cut(df_engineered['Platelet Count(/cumm)'], 
                                                   bins=[0, 150000, 450000, np.inf], 
                                                   labels=[0, 1, 2])  # Low, Normal, High
        df_engineered['Platelet_Category'] = df_engineered['Platelet_Category'].astype(int)
        
        # Combined test results score
        test_columns = ['Bone Marrow Aspiration(Positive / Negative / Not Done)', 
                       'Lymph Node Biopsy(Positive / Negative / Not Done)', 
                       'Lumbar Puncture(Spinal Tap)']
        df_engineered['Positive_Tests_Count'] = (df_engineered[test_columns] == 1).sum(axis=1)
        df_engineered['Negative_Tests_Count'] = (df_engineered[test_columns] == 0).sum(axis=1)
        df_engineered['Not_Done_Tests_Count'] = (df_engineered[test_columns] == -1).sum(axis=1)
        df_engineered['Test_Completion_Rate'] = (df_engineered[test_columns] != -1).sum(axis=1) / len(test_columns)
        
        # Advanced risk scoring
        df_engineered['Clinical_Risk_Score'] = (
            (df_engineered['Age'] / 100) * 0.2 +
            df_engineered['Positive_Tests_Count'] * 0.3 +
            (df_engineered['WBC_Category'] / 3) * 0.25 +
            (df_engineered['Platelet_Category'] / 2) * 0.15 +
            (df_engineered['Genetic_Data(BCR-ABL, FLT3)'] / 2) * 0.1
        )
        
        # Blood count abnormality indicators
        df_engineered['WBC_Abnormal'] = ((df_engineered['Total WBC count(/cumm)'] < 4000) | 
                                        (df_engineered['Total WBC count(/cumm)'] > 11000)).astype(int)
        df_engineered['Platelet_Abnormal'] = ((df_engineered['Platelet Count(/cumm)'] < 150000) | 
                                             (df_engineered['Platelet Count(/cumm)'] > 450000)).astype(int)
        df_engineered['Both_Counts_Abnormal'] = (df_engineered['WBC_Abnormal'] & df_engineered['Platelet_Abnormal']).astype(int)
        
        return df_engineered
    
    def apply_feature_scaling(self, X_train, X_test):
        """Apply feature scaling to numerical features"""
        numerical_features = ['Age', 'Total WBC count(/cumm)', 'Platelet Count(/cumm)', 
                            'WBC_Platelet_Ratio', 'Age_WBC_Interaction', 'Age_Platelet_Interaction', 
                            'Log_WBC', 'Log_Platelet', 'Sqrt_Age', 'Clinical_Risk_Score']
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Only scale if features exist
        existing_features = [f for f in numerical_features if f in X_train.columns]
        
        if existing_features:
            X_train_scaled[existing_features] = self.scaler.fit_transform(X_train[existing_features])
            X_test_scaled[existing_features] = self.scaler.transform(X_test[existing_features])
        
        return X_train_scaled, X_test_scaled

def comprehensive_feature_selection(X, y, feature_names, top_k=15):
    """Apply multiple feature selection techniques and combine results"""
    
    print("=== COMPREHENSIVE FEATURE SELECTION ===")
    
    # 1. Variance Threshold
    print("\n1. Variance Threshold Selection:")
    variance_selector = VarianceThreshold(threshold=0.01)
    X_variance = variance_selector.fit_transform(X)
    variance_features = np.array(feature_names)[variance_selector.get_support()]
    print(f"Features selected: {len(variance_features)}/{len(feature_names)}")
    
    # 2. Statistical tests
    print("\n2. Statistical Feature Selection:")
    
    # F-test
    k_features = min(top_k, X.shape[1])
    f_selector = SelectKBest(score_func=f_classif, k=k_features)
    X_f = f_selector.fit_transform(X, y)
    f_features = np.array(feature_names)[f_selector.get_support()]
    f_scores = f_selector.scores_
    
    print(f"F-test selected top {k_features} features")
    
    # Mutual Information
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
    X_mi = mi_selector.fit_transform(X, y)
    mi_features = np.array(feature_names)[mi_selector.get_support()]
    mi_scores = mi_selector.scores_
    
    print(f"Mutual Information selected top {k_features} features")
    
    # 3. Model-based selection with XGBoost (fixed parameters)
    print("\n3. XGBoost-based Feature Selection:")
    xgb_selector_model = xgb.XGBClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1
    )
    xgb_selector = SelectFromModel(xgb_selector_model, threshold='median')
    X_xgb = xgb_selector.fit_transform(X, y)
    xgb_features = np.array(feature_names)[xgb_selector.get_support()]
    
    print(f"XGBoost selected {len(xgb_features)} features")
    
    # 4. Recursive Feature Elimination
    print("\n4. Recursive Feature Elimination:")
    rfe_model = xgb.XGBClassifier(
        random_state=42,
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1
    )
    rfe_selector = RFE(rfe_model, n_features_to_select=min(12, X.shape[1]))
    X_rfe = rfe_selector.fit_transform(X, y)
    rfe_features = np.array(feature_names)[rfe_selector.get_support()]
    
    print(f"RFE selected {len(rfe_features)} features")
    
    # Combine all selected features
    all_selected = set(variance_features) | set(f_features) | set(mi_features) | set(xgb_features) | set(rfe_features)
    
    print(f"\n=== FINAL COMBINED FEATURE SET ===")
    print(f"Total unique features selected: {len(all_selected)}")
    print(f"Final features: {sorted(list(all_selected))}")
    
    # Create feature importance summary
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'F_Score': f_scores,
        'MI_Score': mi_scores,
        'Selected_by_Variance': [f in variance_features for f in feature_names],
        'Selected_by_F': [f in f_features for f in feature_names],
        'Selected_by_MI': [f in mi_features for f in feature_names],
        'Selected_by_XGB': [f in xgb_features for f in feature_names],
        'Selected_by_RFE': [f in rfe_features for f in feature_names]
    })
    
    feature_importance_df['Total_Selections'] = (
        feature_importance_df['Selected_by_Variance'].astype(int) +
        feature_importance_df['Selected_by_F'].astype(int) +
        feature_importance_df['Selected_by_MI'].astype(int) +
        feature_importance_df['Selected_by_XGB'].astype(int) +
        feature_importance_df['Selected_by_RFE'].astype(int)
    )
    
    feature_importance_df = feature_importance_df.sort_values('Total_Selections', ascending=False)
    
    return list(all_selected), feature_importance_df

def optimize_xgboost_model(X_train, y_train, cv_folds=5):
    """Optimize XGBoost hyperparameters using GridSearchCV - FIXED VERSION"""
    
    print("=== XGBOOST HYPERPARAMETER OPTIMIZATION ===")
    
    # Simplified parameter grid for faster execution
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 1.5]
    }
    
    # Create XGBoost classifier - REMOVED use_label_encoder parameter
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        tree_method='hist',  # Faster training
        enable_categorical=False
    )
    
    # Stratified K-Fold for better validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Grid search with cross-validation
    print("Performing Grid Search Cross-Validation...")
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0  # Reduced verbosity
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"Grid search completed in {end_time - start_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model_comprehensive(model, X_train, X_test, y_train, y_test, class_names):
    """Comprehensive model evaluation with multiple metrics"""
    
    print("=== COMPREHENSIVE MODEL EVALUATION ===")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    
    # Basic metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\n=== ACCURACY SCORES ===")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print(f"Overfitting Check: {train_accuracy - test_accuracy:.4f}")
    
    # Additional metrics
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    
    # Detailed classification report
    print(f"\n=== CLASSIFICATION REPORT ===")
    available_classes = sorted(np.unique(np.concatenate([y_test, y_test_pred])))
    available_class_names = [class_names[i] for i in available_classes]
    print(classification_report(y_test, y_test_pred, 
                              labels=available_classes,
                              target_names=available_class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=available_class_names, yticklabels=available_class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n=== TOP 10 FEATURE IMPORTANCE ===")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title('Top 15 Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n=== CROSS-VALIDATION RESULTS ===")
    print(f"CV Scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Score')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation Score')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'feature_importance': feature_importance,
        'confusion_matrix': cm,
        'cv_scores': cv_scores
    }

def main():
    """Main function to run the complete ML pipeline"""
    
    print("=== ADVANCED BLOOD CANCER ML PIPELINE (FIXED VERSION) ===")
    
    # Load the processed dataset
    print("\n1. Loading processed dataset...")
    try:
        df = pd.read_csv('public/sample_blood_cancer_diseases_dataset_numeric.csv')
        print(f"✅ Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Dataset columns: {list(df.columns)}")
    except FileNotFoundError:
        print("❌ Error: Could not find the processed dataset file.")
        print("Please ensure 'public/sample_blood_cancer_diseases_dataset_numeric.csv' exists.")
        return None, None, None
    
    # Display basic info
    print(f"\nTarget distribution:")
    target_counts = df['Cancer_Type(AML, ALL, CLL)'].value_counts().sort_index()
    print(target_counts)
    
    # Define class names for interpretability
    class_names = ['AML', 'ALL', 'CLL', 'Lymphoma', 'Multiple Myeloma', 'CML']
    
    # Feature Engineering
    print("\n2. Applying Advanced Feature Engineering...")
    feature_engineer = AdvancedFeatureEngineering()
    df_engineered = feature_engineer.create_interaction_features(df)
    
    print(f"Original features: {df.shape[1]}")
    print(f"After feature engineering: {df_engineered.shape[1]}")
    new_features = [col for col in df_engineered.columns if col not in df.columns]
    print(f"New features added: {new_features}")
    
    # Prepare features and target
    X = df_engineered.drop('Cancer_Type(AML, ALL, CLL)', axis=1)
    y = df_engineered['Cancer_Type(AML, ALL, CLL)']
    
    print(f"\nFinal feature set ({len(X.columns)} features):")
    for i, col in enumerate(X.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Split the data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    print(f"Training target distribution: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Testing target distribution: {y_test.value_counts().sort_index().to_dict()}")
    
    # Apply feature scaling
    print("\n4. Applying feature scaling...")
    X_train_scaled, X_test_scaled = feature_engineer.apply_feature_scaling(X_train, X_test)
    print("✅ Feature scaling completed")
    
    # Feature Selection
    print("\n5. Performing feature selection...")
    selected_features, feature_importance_df = comprehensive_feature_selection(
        X_train_scaled.values, y_train.values, X_train_scaled.columns.tolist()
    )
    
    # Filter to selected features
    X_train_selected = X_train_scaled[selected_features]
    X_test_selected = X_test_scaled[selected_features]
    
    print(f"\n✅ Using {len(selected_features)} selected features for modeling")
    
    # Save feature selection results
    try:
        feature_importance_df.to_csv('public/feature_selection_results.csv', index=False)
        print("✅ Feature selection results saved to: public/feature_selection_results.csv")
    except Exception as e:
        print(f"⚠️ Warning: Could not save feature selection results: {e}")
    
    # Model Training and Optimization
    print("\n6. Training and optimizing XGBoost model...")
    try:
        best_model, best_params = optimize_xgboost_model(X_train_selected, y_train)
        print("✅ Model optimization completed successfully!")
    except Exception as e:
        print(f"❌ Error in model optimization: {e}")
        return None, None, None
    
    # Model Evaluation
    print("\n7. Comprehensive model evaluation...")
    try:
        evaluation_results = evaluate_model_comprehensive(
            best_model, X_train_selected, X_test_selected, y_train, y_test, class_names
        )
        print("✅ Model evaluation completed!")
    except Exception as e:
        print(f"❌ Error in model evaluation: {e}")
        return None, None, None
    
    # Save the trained model
    print("\n8. Saving the model...")
    try:
        # Save using joblib (more reliable for scikit-learn models)
        joblib.dump(best_model, 'public/xgboost_blood_cancer_model.pkl')
        print("✅ Model saved to: public/xgboost_blood_cancer_model.pkl")
        
        # Also save feature names and selected features
        model_info = {
            'selected_features': selected_features,
            'class_names': class_names,
            'feature_scaler': feature_engineer.scaler,
            'best_params': best_params
        }
        joblib.dump(model_info, 'public/model_info.pkl')
        print("✅ Model info saved to: public/model_info.pkl")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not save model: {e}")
    
    # Save predictions
    print("\n9. Saving predictions...")
    try:
        test_predictions = best_model.predict(X_test_selected)
        test_probabilities = best_model.predict_proba(X_test_selected)
        
        results_df = pd.DataFrame({
            'True_Label': y_test.values,
            'Predicted_Label': test_predictions,
            'True_Cancer_Type': [class_names[i] if i < len(class_names) else f'Unknown_{i}' for i in y_test.values],
            'Predicted_Cancer_Type': [class_names[i] if i < len(class_names) else f'Unknown_{i}' for i in test_predictions],
            'Correct_Prediction': (y_test.values == test_predictions).astype(int)
        })
        
        # Add probability columns for available classes
        for i in range(test_probabilities.shape[1]):
            class_name = class_names[i] if i < len(class_names) else f'Class_{i}'
            results_df[f'Prob_{class_name}'] = test_probabilities[:, i]
        
        results_df.to_csv('public/model_predictions.csv', index=False)
        print("✅ Predictions saved to: public/model_predictions.csv")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not save predictions: {e}")
    
    # Final Summary
    print("\n" + "="*60)
    print("🎯 FINAL MODEL SUMMARY")
    print("="*60)
    print(f"📊 Best Parameters: {best_params}")
    print(f"🎯 Test Accuracy: {evaluation_results['test_accuracy']:.4f}")
    print(f"📈 Cross-validation Score: {evaluation_results['cv_scores'].mean():.4f} (+/- {evaluation_results['cv_scores'].std() * 2:.4f})")
    print(f"🔧 Selected Features: {len(selected_features)}")
    print(f"🏆 Top 5 Important Features:")
    for i, (_, row) in enumerate(evaluation_results['feature_importance'].head(5).iterrows()):
        print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    print(f"\n📁 Generated Files:")
    print(f"  - public/xgboost_blood_cancer_model.pkl")
    print(f"  - public/model_info.pkl")
    print(f"  - public/feature_selection_results.csv")
    print(f"  - public/model_predictions.csv")
    print("\n✅ Pipeline completed successfully!")
    
    return best_model, evaluation_results, selected_features

if __name__ == "__main__":
    model, results, features = main()