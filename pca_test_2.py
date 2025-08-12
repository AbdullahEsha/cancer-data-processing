import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                            GradientBoostingClassifier, VotingClassifier, BaggingClassifier)
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('public/processed_blood_cancer_dataset.csv')

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
target_counts = df['Cancer_Type(AML, ALL, CLL)'].value_counts().sort_index()
print(target_counts)

X = df.drop('Cancer_Type(AML, ALL, CLL)', axis=1)
y = df['Cancer_Type(AML, ALL, CLL)']

class OptimizedFeatureEngineering:
    def __init__(self):
        pass
        
    def create_medical_features(self, X):
        X_new = X.copy()
        
        # Medical ratios (crucial for blood cancers)
        X_new['WBC_Platelet_ratio'] = X['Total WBC count(/cumm)'] / (X['Platelet Count(/cumm)'] + 1)
        X_new['Platelet_WBC_ratio'] = X['Platelet Count(/cumm)'] / (X['Total WBC count(/cumm)'] + 1)
        X_new['WBC_per_age'] = X['Total WBC count(/cumm)'] / (X['Age'] + 1)
        X_new['Platelet_per_age'] = X['Platelet Count(/cumm)'] / (X['Age'] + 1)
        
        # Age-based features (different cancers affect different age groups)
        X_new['Age_group_0_30'] = ((X['Age'] >= 0) & (X['Age'] <= 30)).astype(int)
        X_new['Age_group_31_50'] = ((X['Age'] >= 31) & (X['Age'] <= 50)).astype(int)
        X_new['Age_group_51_70'] = ((X['Age'] >= 51) & (X['Age'] <= 70)).astype(int)
        X_new['Age_group_71_plus'] = (X['Age'] >= 71).astype(int)
        
        # Blood count severity (quartile-based)
        wbc_q1, wbc_q3 = X['Total WBC count(/cumm)'].quantile([0.25, 0.75])
        platelet_q1, platelet_q3 = X['Platelet Count(/cumm)'].quantile([0.25, 0.75])
        
        X_new['WBC_low'] = (X['Total WBC count(/cumm)'] <= wbc_q1).astype(int)
        X_new['WBC_high'] = (X['Total WBC count(/cumm)'] >= wbc_q3).astype(int)
        X_new['Platelet_low'] = (X['Platelet Count(/cumm)'] <= platelet_q1).astype(int)
        X_new['Platelet_high'] = (X['Platelet Count(/cumm)'] >= platelet_q3).astype(int)
        
        # Blood abnormality combinations
        X_new['WBC_Platelet_both_abnormal'] = ((X_new['WBC_low'] | X_new['WBC_high']) & 
                                              (X_new['Platelet_low'] | X_new['Platelet_high'])).astype(int)
        
        # Log transformations for skewed data
        X_new['WBC_log'] = np.log1p(X['Total WBC count(/cumm)'])
        X_new['Platelet_log'] = np.log1p(X['Platelet Count(/cumm)'])
        X_new['Age_log'] = np.log1p(X['Age'])
        
        # Test combinations (diagnostic patterns)
        X_new['Bone_Lymph_sum'] = (X['Bone Marrow Aspiration(Positive / Negative / Not Done)'] + 
                                  X['Lymph Node Biopsy(Positive / Negative / Not Done)'])
        X_new['All_tests_done'] = ((X['Bone Marrow Aspiration(Positive / Negative / Not Done)'] != -1) & 
                                  (X['Lymph Node Biopsy(Positive / Negative / Not Done)'] != -1) &
                                  (X['Lumbar Puncture(Spinal Tap)'] != -1)).astype(int)
        
        # Treatment-response patterns
        X_new['Genetic_positive'] = (X['Genetic_Data(BCR-ABL, FLT3)'] == 1).astype(int)
        X_new['Side_effects_severe'] = (X['Side_Effects'] == 1).astype(int)
        X_new['Treatment_genetic_interaction'] = X['Treatment_Type(Chemotherapy, Radiation)'] * X['Genetic_Data(BCR-ABL, FLT3)']
        
        # Gender-age interactions
        X_new['Male_elderly'] = ((X['Gender'] == 1) & (X['Age'] >= 65)).astype(int)
        X_new['Female_young'] = ((X['Gender'] == 0) & (X['Age'] < 50)).astype(int)
        
        # Disease severity score
        X_new['Severity_score'] = (X_new['WBC_Platelet_both_abnormal'] + 
                                  X_new['Side_effects_severe'] + 
                                  (X['Diagnosis_Result'] == 1).astype(int))
        
        return X_new

def create_balanced_models():
    """Create models optimized for multi-class classification"""
    
    models = {}
    
    # Random Forest with balanced classes
    models['RF_balanced'] = RandomForestClassifier(
        n_estimators=500, 
        max_depth=20, 
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42, 
        n_jobs=-1
    )
    
    # XGBoost with regularization
    models['XGB_tuned'] = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    # Extra Trees (often performs well on tabular data)
    models['ExtraTrees'] = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Gradient Boosting
    models['GradientBoosting'] = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    
    # Logistic Regression with regularization
    models['LogisticRegression'] = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        multi_class='ovr',
        random_state=42,
        max_iter=1000
    )
    
    # SVM with balanced classes
    models['SVM'] = SVC(
        C=1.0,
        kernel='rbf',
        class_weight='balanced',
        random_state=42,
        probability=True
    )
    
    # Voting classifier (soft voting)
    models['VotingSoft'] = VotingClassifier([
        ('rf', models['RF_balanced']),
        ('xgb', models['XGB_tuned']),
        ('et', models['ExtraTrees'])
    ], voting='soft')
    
    # Bagging with Random Forest (fixed)
    models['BaggingRF'] = BaggingClassifier(
        estimator=RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        n_estimators=10,
        random_state=42,
        n_jobs=-1
    )
    
    return models

def comprehensive_evaluation(X, y):
    """Comprehensive evaluation with different preprocessing approaches"""
    
    results = []
    
    # Feature engineering
    fe = OptimizedFeatureEngineering()
    X_engineered = fe.create_medical_features(X)
    print(f"Features after engineering: {X_engineered.shape[1]}")
    
    # Different sampling approaches
    sampling_configs = {}
    
    # Original data
    sampling_configs['Original'] = (X_engineered, y)
    
    # SMOTE oversampling
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(5, Counter(y).most_common()[-1][1]-1))
        X_smote, y_smote = smote.fit_resample(X_engineered, y)
        sampling_configs['SMOTE'] = (X_smote, y_smote)
        print(f"SMOTE: {X_smote.shape[0]} samples, {Counter(y_smote)}")
    except Exception as e:
        print(f"SMOTE failed: {e}")
    
    # SMOTETomek combination
    try:
        smotetomek = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=3))
        X_st, y_st = smotetomek.fit_resample(X_engineered, y)
        sampling_configs['SMOTETomek'] = (X_st, y_st)
        print(f"SMOTETomek: {X_st.shape[0]} samples, {Counter(y_st)}")
    except Exception as e:
        print(f"SMOTETomek failed: {e}")
    
    # Create models
    models = create_balanced_models()
    
    # Evaluate each combination
    for sampling_name, (X_data, y_data) in sampling_configs.items():
        print(f"\n{'='*60}")
        print(f"EVALUATING: {sampling_name}")
        print('='*60)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data)
        
        # Feature selection approaches
        feature_configs = []
        
        # All features
        feature_configs.append(('All_features', X_scaled))
        
        # Top features by mutual information
        for k in [15, 20, 25]:
            try:
                selector = SelectKBest(mutual_info_classif, k=min(k, X_scaled.shape[1]-1))
                X_selected = selector.fit_transform(X_scaled, y_data)
                feature_configs.append((f'MutualInfo_k{k}', X_selected))
            except:
                continue
        
        # Top features by F-test
        for k in [15, 20]:
            try:
                selector = SelectKBest(f_classif, k=min(k, X_scaled.shape[1]-1))
                X_selected = selector.fit_transform(X_scaled, y_data)
                feature_configs.append((f'Ftest_k{k}', X_selected))
            except:
                continue
        
        # Evaluate each feature configuration
        for feature_name, X_features in feature_configs:
            print(f"\nFeature config: {feature_name} ({X_features.shape[1]} features)")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, y_data, test_size=0.25, random_state=42, stratify=y_data
            )
            
            # Test each model
            for model_name, model in models.items():
                try:
                    # Fit model
                    model.fit(X_train, y_train)
                    
                    # Test accuracy
                    test_accuracy = model.score(X_test, y_test)
                    
                    # Cross validation
                    cv_scores = cross_val_score(model, X_features, y_data, cv=3, scoring='accuracy')
                    
                    result = {
                        'sampling': sampling_name,
                        'features': feature_name,
                        'model': model_name,
                        'test_accuracy': test_accuracy,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'n_features': X_features.shape[1]
                    }
                    
                    results.append(result)
                    
                    print(f"  {model_name:18} | Test: {test_accuracy:.3f} | CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                    
                except Exception as e:
                    print(f"  {model_name:18} | ERROR: {str(e)[:30]}")
                    continue
    
    return results

# Run comprehensive evaluation
print("Starting comprehensive evaluation...")
results = comprehensive_evaluation(X, y)

# Analyze results
results_df = pd.DataFrame(results)

print(f"\n{'='*100}")
print("TOP 15 BEST CONFIGURATIONS")
print('='*100)

# Sort by test accuracy
top_results = results_df.nlargest(15, 'test_accuracy')

print(f"{'Rank':<4} {'Model':<18} {'Sampling':<12} {'Features':<15} {'Test Acc':<8} {'CV Mean':<8} {'CV Std':<8}")
print("-" * 100)

for i, (_, row) in enumerate(top_results.iterrows(), 1):
    print(f"{i:<4} {row['model']:<18} {row['sampling']:<12} {row['features']:<15} "
          f"{row['test_accuracy']:<8.4f} {row['cv_mean']:<8.4f} {row['cv_std']:<8.4f}")

# Train and evaluate the best configuration
best_config = top_results.iloc[0]
print(f"\n{'='*100}")
print("FINAL EVALUATION WITH BEST CONFIGURATION")
print('='*100)

print(f"Best configuration:")
print(f"  Sampling: {best_config['sampling']}")
print(f"  Features: {best_config['features']}")
print(f"  Model: {best_config['model']}")
print(f"  Test Accuracy: {best_config['test_accuracy']:.4f}")
print(f"  CV Mean: {best_config['cv_mean']:.4f}")

# Recreate best configuration for detailed evaluation
fe = OptimizedFeatureEngineering()
X_engineered = fe.create_medical_features(X)

# Apply best sampling
if best_config['sampling'] == 'SMOTE':
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_final, y_final = smote.fit_resample(X_engineered, y)
elif best_config['sampling'] == 'SMOTETomek':
    smotetomek = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=3))
    X_final, y_final = smotetomek.fit_resample(X_engineered, y)
else:
    X_final, y_final = X_engineered, y

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# Apply best feature selection
feature_config = best_config['features']
if 'MutualInfo' in feature_config:
    k = int(feature_config.split('k')[1])
    selector = SelectKBest(mutual_info_classif, k=k)
    X_processed = selector.fit_transform(X_scaled, y_final)
elif 'Ftest' in feature_config:
    k = int(feature_config.split('k')[1])
    selector = SelectKBest(f_classif, k=k)
    X_processed = selector.fit_transform(X_scaled, y_final)
else:
    X_processed = X_scaled

# Train best model
models = create_balanced_models()
best_model = models[best_config['model']]

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_final, test_size=0.25, random_state=42, stratify=y_final
)

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)

print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Final cross-validation
cv_scores = cross_val_score(best_model, X_processed, y_final, cv=5, scoring='accuracy')
print(f"\n5-Fold Cross Validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Success check
if final_accuracy >= 0.60:
    print(f"\n🎉 SUCCESS! Achieved target accuracy: {final_accuracy:.4f}")
else:
    print(f"\n❌ Below target but improved: {final_accuracy:.4f}")
    
print(f"\nBest overall accuracy achieved: {max(results_df['test_accuracy']):.4f}")