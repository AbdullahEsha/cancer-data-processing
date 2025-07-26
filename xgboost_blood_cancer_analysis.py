import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, RobustScaler
from sklearn.experimental import enable_iterative_imputer  # Enable experimental imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
import json
import os
import joblib
from scipy import stats
warnings.filterwarnings('ignore')

if not os.path.exists("public/blood_cancer_xgboost_results"):
    os.makedirs("public/blood_cancer_xgboost_results")

class EnhancedBloodCancerClassifier:
    def __init__(self):
        self.scaler = RobustScaler()  # More robust to outliers
        self.label_encoder = LabelEncoder()
        self.imputer = IterativeImputer(random_state=42)  # More sophisticated imputation
        self.feature_selector = None
        self.best_model = None
        self.feature_names = None
        self.model_performance = {}
        self.poly_features = None
        self.sampler = None
        self.numeric_features = []
        self.categorical_features = []
        
    def load_and_explore_data(self, file_path):
        """Load and explore the dataset with advanced analysis"""
        print("=" * 60)
        print("LOADING AND EXPLORING DATASET")
        print("=" * 60)
        
        # Load data
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Basic info
        print("\nDataset Info:")
        print(df.info())
        
        # Missing values analysis
        print("\nMissing Values Analysis:")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_data, 
            'Percentage': missing_percent
        }).sort_values('Percentage', ascending=False)
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Statistical summary
        print("\nStatistical Summary:")
        print(df.describe())
        
        # Target distribution
        target_col = None
        for col in df.columns:
            if 'cancer' in col.lower() and 'type' in col.lower():
                target_col = col
                break
        
        if target_col:
            print(f"\nTarget Variable Distribution ({target_col}):")
            target_counts = df[target_col].value_counts()
            print(target_counts)
            print(f"Class balance ratio: {target_counts.min() / target_counts.max():.3f}")
            
            # Enhanced visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Target distribution
            target_counts.plot(kind='bar', ax=axes[0,0], color='skyblue')
            axes[0,0].set_title('Cancer Type Distribution')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Missing values heatmap
            if missing_df['Missing Count'].sum() > 0:
                missing_matrix = df.isnull()
                sns.heatmap(missing_matrix, ax=axes[0,1], cbar=True, yticklabels=False)
                axes[0,1].set_title('Missing Values Pattern')
            else:
                axes[0,1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[0,1].transAxes)
                axes[0,1].set_title('Missing Values Pattern')
            
            # Correlation matrix for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1,0])
                axes[1,0].set_title('Feature Correlation Matrix')
            else:
                axes[1,0].text(0.5, 0.5, 'No Numeric Features for Correlation', ha='center', va='center', transform=axes[1,0].transAxes)
                axes[1,0].set_title('Feature Correlation Matrix')
            
            # Class distribution pie chart
            target_counts.plot(kind='pie', ax=axes[1,1], autopct='%1.1f%%')
            axes[1,1].set_title('Cancer Type Proportion')
            
            plt.tight_layout()
            plt.savefig("public/blood_cancer_xgboost_results/data_exploration.png", dpi=300, bbox_inches='tight')
            plt.show()

        return df
    
    def advanced_preprocessing(self, df):
        """Ultra-advanced data preprocessing with feature engineering"""
        print("\n" + "=" * 60)
        print("ADVANCED DATA PREPROCESSING & FEATURE ENGINEERING")
        print("=" * 60)
        
        df_processed = df.copy()
        
        # Identify target column
        target_col = None
        for col in df.columns:
            if 'cancer' in col.lower() and 'type' in col.lower():
                target_col = col
                break
        
        if not target_col:
            raise ValueError("Target column (Cancer Type) not found!")
        
        print(f"Target column identified: {target_col}")
        
        # Separate features and target
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        # Advanced target cleaning and consolidation
        y = y.astype(str).str.strip().str.upper()
        
        print(f"Original target distribution:")
        print(y.value_counts())
        
        # Intelligent cancer type mapping based on medical classification
        cancer_mapping = {
            'LYMPHOMA': 'CLL',
            'MULTIPLE MYELOMA': 'CLL',
            'CHRONIC LYMPHOCYTIC LEUKEMIA': 'CLL',
            'ACUTE MYELOID LEUKEMIA': 'AML',
            'ACUTE LYMPHOBLASTIC LEUKEMIA': 'ALL',
            'CHRONIC MYELOID LEUKEMIA': 'AML',
            'MYELODYSPLASTIC SYNDROME': 'AML',
            'HODGKIN LYMPHOMA': 'CLL',
            'NON-HODGKIN LYMPHOMA': 'CLL',
            'ACUTE LYMPHOCYTIC LEUKEMIA': 'ALL',
            'CHRONIC LYMPHOID LEUKEMIA': 'CLL'
        }
        y = y.replace(cancer_mapping)
        
        # Handle rare classes
        value_counts = y.value_counts()
        print(f"Target distribution after mapping:")
        print(value_counts)
        
        # Combine very rare classes (< 1% of data)
        total_samples = len(y)
        rare_threshold = max(3, int(0.01 * total_samples))  # At least 1% or 3 samples
        rare_classes = value_counts[value_counts < rare_threshold].index.tolist()
        
        if rare_classes:
            print(f"Rare classes found (< {rare_threshold} samples): {rare_classes}")
            most_common_class = value_counts.index[0]
            for rare_class in rare_classes:
                print(f"Combining {rare_class} with {most_common_class}")
                y = y.replace(rare_class, most_common_class)
        
        final_value_counts = y.value_counts()
        print(f"Final target distribution:")
        print(final_value_counts)
        
        # Remove singleton classes
        classes_to_keep = final_value_counts[final_value_counts >= 2].index.tolist()
        if len(classes_to_keep) != len(final_value_counts):
            mask = y.isin(classes_to_keep)
            X = X[mask]
            y = y[mask]
            print(f"Final samples after cleanup: {len(y)}")
        
        # Advanced feature processing - Fixed feature detection
        numeric_features = []
        categorical_features = []
        
        for col in X.columns:
            # More intelligent numeric detection
            if X[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
            else:
                # Try to convert to numeric
                try:
                    # Handle various numeric formats
                    temp_series = X[col].astype(str).str.replace(',', '').str.replace('(', '').str.replace(')', '')
                    temp_series = temp_series.str.replace('>', '').str.replace('<', '').str.replace('=', '')
                    temp_series = temp_series.str.replace('nan', '').str.replace('NaN', '').str.replace('', '0')
                    
                    # Try to convert to numeric
                    numeric_series = pd.to_numeric(temp_series, errors='coerce')
                    
                    # Check if most values are numeric (at least 70%)
                    non_null_ratio = numeric_series.notna().sum() / len(numeric_series)
                    if non_null_ratio >= 0.7:
                        X[col] = numeric_series
                        numeric_features.append(col)
                    else:
                        categorical_features.append(col)
                except:
                    categorical_features.append(col)
        
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
        print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
        
        # Advanced numeric feature processing
        X_numeric = pd.DataFrame()
        if numeric_features:
            for col in numeric_features:
                # Advanced cleaning
                if X[col].dtype == 'object':
                    X[col] = X[col].astype(str)
                    X[col] = X[col].str.replace(',', '').str.replace('(', '').str.replace(')', '')
                    X[col] = X[col].str.replace('>', '').str.replace('<', '').str.replace('=', '')
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                
                # Fill missing values with median
                X[col] = X[col].fillna(X[col].median())
                
                # Advanced outlier handling using IQR with medical context
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # More conservative outlier bounds for medical data
                if IQR > 0:
                    lower_bound = Q1 - 2.0 * IQR  # Less aggressive
                    upper_bound = Q3 + 2.0 * IQR
                    
                    # Cap outliers instead of removing
                    X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
                
                # Log transformation for skewed features
                if X[col].min() > 0:  # Only for positive values
                    skewness = stats.skew(X[col].dropna())
                    if abs(skewness) > 1:  # Highly skewed
                        X[f'{col}_log'] = np.log1p(X[col])
                        print(f"Created log-transformed feature: {col}_log")
                        numeric_features.append(f'{col}_log')  # Add to numeric features list
            
            X_numeric = X[numeric_features].copy()
            
            # Advanced imputation only if there are still missing values
            if X_numeric.isnull().sum().sum() > 0:
                print("Performing advanced iterative imputation...")
                X_numeric = pd.DataFrame(
                    self.imputer.fit_transform(X_numeric),
                    columns=numeric_features,
                    index=X.index
                )
            
            # Robust scaling
            X_numeric = pd.DataFrame(
                self.scaler.fit_transform(X_numeric),
                columns=numeric_features,
                index=X.index
            )
        
        # Advanced categorical processing
        X_categorical = pd.DataFrame(index=X.index)
        if categorical_features:
            for col in categorical_features:
                # Clean categorical data
                X[col] = X[col].astype(str).str.strip().str.title()
                X[col] = X[col].replace(['Nan', 'None', 'null', ''], 'Unknown')
                X[col] = X[col].fillna('Unknown')
                
                # Handle high cardinality categorical features
                value_counts = X[col].value_counts()
                if len(value_counts) > 10:  # High cardinality
                    # Keep only top categories, group others as 'Other'
                    top_categories = value_counts.head(8).index.tolist()
                    X[col] = X[col].apply(lambda x: x if x in top_categories else 'Other')
                
                # One-hot encoding
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X_categorical = pd.concat([X_categorical, dummies], axis=1)
        
        # Combine features
        if not X_numeric.empty and not X_categorical.empty:
            X_processed = pd.concat([X_numeric, X_categorical], axis=1)
        elif not X_numeric.empty:
            X_processed = X_numeric
        else:
            X_processed = X_categorical
        
        # Advanced feature engineering - FIXED VERSION
        print(f"\nCreating advanced engineered features...")
        
        # Only create interaction features for numeric columns
        if len(numeric_features) >= 2:
            # Take first few numeric features for interactions to avoid explosion
            selected_numeric = numeric_features[:min(4, len(numeric_features))]
            
            for i in range(len(selected_numeric)):
                for j in range(i+1, len(selected_numeric)):
                    col1, col2 = selected_numeric[i], selected_numeric[j]
                    if col1 in X_processed.columns and col2 in X_processed.columns:
                        # Check if both columns are numeric (not boolean)
                        if (X_processed[col1].dtype in ['int64', 'float64'] and 
                            X_processed[col2].dtype in ['int64', 'float64']):
                            
                            # Multiplicative interaction
                            X_processed[f'{col1}_x_{col2}'] = X_processed[col1] * X_processed[col2]
                            
                            # Ratio feature (with safe division)
                            X_processed[f'{col1}_ratio_{col2}'] = X_processed[col1] / (X_processed[col2] + 1e-8)
                            
                            # Difference feature
                            X_processed[f'{col1}_diff_{col2}'] = X_processed[col1] - X_processed[col2]
                            
                            print(f"Created interaction features for {col1} and {col2}")
        
        # Polynomial features for numeric features only
        if len(numeric_features) >= 2:
            # Select only original numeric features (not log-transformed ones)
            original_numeric = [col for col in numeric_features if not col.endswith('_log')][:3]  # Limit to 3
            available_poly_cols = [col for col in original_numeric if col in X_processed.columns]
            
            if len(available_poly_cols) >= 2:
                try:
                    self.poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                    poly_features_array = self.poly_features.fit_transform(X_processed[available_poly_cols])
                    
                    # Get feature names
                    poly_feature_names = self.poly_features.get_feature_names_out(available_poly_cols)
                    
                    # Add only the interaction terms (not the original features)
                    for i, name in enumerate(poly_feature_names):
                        if ' ' in name:  # Interaction terms contain spaces
                            clean_name = name.replace(' ', '_x_')
                            if clean_name not in X_processed.columns:
                                X_processed[f'poly_{clean_name}'] = poly_features_array[:, i]
                    print(f"Created polynomial interaction features")
                except Exception as e:
                    print(f"Polynomial feature creation failed: {e}")
        
        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\nFinal processed dataset:")
        print(f"Shape: {X_processed.shape}")
        print(f"Features: {X_processed.shape[1]}")
        print(f"Samples: {X_processed.shape[0]}")
        
        self.feature_names = X_processed.columns.tolist()
        
        return X_processed, y_encoded, y
    
    def advanced_feature_selection(self, X, y):
        """Multi-stage advanced feature selection"""
        print("\n" + "=" * 60)
        print("ADVANCED FEATURE SELECTION")
        print("=" * 60)
        
        print(f"Starting with {X.shape[1]} features")
        
        # Stage 1: Remove low variance features
        from sklearn.feature_selection import VarianceThreshold
        var_threshold = VarianceThreshold(threshold=0.005)  # More conservative
        X_var = var_threshold.fit_transform(X)
        selected_features_var = np.array(self.feature_names)[var_threshold.get_support()]
        print(f"After variance threshold: {X_var.shape[1]} features")
        
        # Stage 2: Univariate selection with multiple methods
        # Method 1: F-statistics
        k_best_f = min(60, X_var.shape[1])
        selector_f = SelectKBest(score_func=f_classif, k=k_best_f)
        X_selected_f = selector_f.fit_transform(X_var, y)
        
        # Method 2: Mutual information
        k_best_mi = min(60, X_var.shape[1])
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=k_best_mi)
        X_selected_mi = selector_mi.fit_transform(X_var, y)
        
        # Combine selections (union of both methods)
        features_f = set(np.where(selector_f.get_support())[0])
        features_mi = set(np.where(selector_mi.get_support())[0])
        combined_features = list(features_f.union(features_mi))
        
        X_combined = X_var[:, combined_features]
        selected_features_combined = selected_features_var[combined_features]
        print(f"After combined univariate selection: {X_combined.shape[1]} features")
        
        # Stage 3: Tree-based feature importance
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_selector.fit(X_combined, y)
        
        # Get feature importances
        importances = rf_selector.feature_importances_
        importance_indices = np.argsort(importances)[::-1]
        
        # Select top features based on importance
        n_important = min(40, len(importance_indices))
        important_indices = importance_indices[:n_important]
        X_important = X_combined[:, important_indices]
        selected_features_important = selected_features_combined[important_indices]
        
        print(f"After importance-based selection: {X_important.shape[1]} features")
        
        # Stage 4: Recursive Feature Elimination with cross-validation
        from sklearn.feature_selection import RFECV
        
        estimator = xgb.XGBClassifier(n_estimators=50, random_state=42, eval_metric='mlogloss')
        n_features_final = min(25, X_important.shape[1])
        
        if X_important.shape[1] > n_features_final:
            rfe = RFE(estimator=estimator, n_features_to_select=n_features_final, step=1)
            X_final = rfe.fit_transform(X_important, y)
            final_features = selected_features_important[rfe.support_]
        else:
            X_final = X_important
            final_features = selected_features_important
            rfe = None
        
        print(f"Final selected features: {X_final.shape[1]}")
        print(f"Selected feature names: {list(final_features)}")
        
        # Show feature importance ranking
        if len(final_features) > 0:
            final_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            final_rf.fit(X_final, y)
            final_importances = final_rf.feature_importances_
            
            feature_importance_df = pd.DataFrame({
                'feature': final_features,
                'importance': final_importances
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 most important features:")
            print(feature_importance_df.head(10))
            
            # Save feature importance plot
            plt.figure(figsize=(10, 8))
            top_features = feature_importance_df.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Most Important Features')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig("public/blood_cancer_xgboost_results/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        self.feature_selector = (var_threshold, selector_f, selector_mi, combined_features, important_indices, rfe)
        
        return X_final, final_features
    
    def handle_class_imbalance(self, X, y):
        """Advanced class imbalance handling"""
        print("\n" + "=" * 60)
        print("HANDLING CLASS IMBALANCE")
        print("=" * 60)
        
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        print(f"Original class distribution: {class_distribution}")
        
        # Calculate imbalance ratio
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = min_count / max_count
        print(f"Imbalance ratio: {imbalance_ratio:.3f}")
        
        if imbalance_ratio < 0.7:  # Imbalanced dataset
            print("Dataset is imbalanced. Applying advanced sampling techniques...")
            
            # Try different sampling strategies
            strategies = {}
            
            # SMOTE
            try:
                k_neighbors = min(5, min_count-1) if min_count > 1 else 1
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_smote, y_smote = smote.fit_resample(X, y)
                strategies['SMOTE'] = (X_smote, y_smote, smote)
                print(f"SMOTE applied. New shape: {X_smote.shape}")
            except Exception as e:
                print(f"SMOTE failed: {e}")
            
            # ADASYN
            try:
                n_neighbors = min(5, min_count-1) if min_count > 1 else 1
                adasyn = ADASYN(random_state=42, n_neighbors=n_neighbors)
                X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
                strategies['ADASYN'] = (X_adasyn, y_adasyn, adasyn)
                print(f"ADASYN applied. New shape: {X_adasyn.shape}")
            except Exception as e:
                print(f"ADASYN failed: {e}")
            
            # SMOTETomek (combination)
            try:
                smote_tomek = SMOTETomek(random_state=42)
                X_smote_tomek, y_smote_tomek = smote_tomek.fit_resample(X, y)
                strategies['SMOTETomek'] = (X_smote_tomek, y_smote_tomek, smote_tomek)
                print(f"SMOTETomek applied. New shape: {X_smote_tomek.shape}")
            except Exception as e:
                print(f"SMOTETomek failed: {e}")
            
            # Choose the best strategy (prefer SMOTE if available)
            if 'SMOTE' in strategies:
                X_resampled, y_resampled, self.sampler = strategies['SMOTE']
                print("Using SMOTE for final resampling")
            elif 'ADASYN' in strategies:
                X_resampled, y_resampled, self.sampler = strategies['ADASYN']
                print("Using ADASYN for final resampling")
            elif 'SMOTETomek' in strategies:
                X_resampled, y_resampled, self.sampler = strategies['SMOTETomek']
                print("Using SMOTETomek for final resampling")
            else:
                print("All sampling strategies failed. Using original data.")
                X_resampled, y_resampled = X, y
            
            # Show new distribution
            unique_new, counts_new = np.unique(y_resampled, return_counts=True)
            new_distribution = dict(zip(unique_new, counts_new))
            print(f"New class distribution: {new_distribution}")
            
            return X_resampled, y_resampled
        else:
            print("Dataset is relatively balanced. No sampling applied.")
            return X, y
    
    def create_advanced_models(self):
        """Create ensemble of optimized models"""
        models = {
            'Optimized_XGBoost': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                eval_metric='mlogloss',
                n_jobs=-1
            ),
            'Optimized_LightGBM': lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            ),
            'Optimized_CatBoost': CatBoostClassifier(
                iterations=300,
                depth=8,
                learning_rate=0.05,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False
            ),
            'Optimized_RandomForest': RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'SVM_RBF': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'Neural_Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.01,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            ),
            'Logistic_Regression': LogisticRegression(
                C=10,
                max_iter=2000,
                random_state=42,
                solver='liblinear',
                n_jobs=-1
            )
        }
        return models
    
    # Continuing from where the code was cut off...

    def hyperparameter_optimization(self, models, X_train, y_train):
        """Optimize hyperparameters for top models"""
        print("\n" + "=" * 60)
        print("HYPERPARAMETER OPTIMIZATION")
        print("=" * 60)
        
        # Quick evaluation to identify top models
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        quick_scores = {}
        
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                quick_scores[name] = scores.mean()
                print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                quick_scores[name] = 0
        
        # Select top 3 models for optimization
        top_models = sorted(quick_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\nTop 3 models selected for optimization: {[model[0] for model in top_models]}")
        
        optimized_models = {}
        
        for model_name, _ in top_models:
            print(f"\nOptimizing {model_name}...")
            
            if model_name == 'Optimized_XGBoost':
                param_grid = {
                    'n_estimators': [200, 300, 400],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9]
                }
                base_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss', n_jobs=-1)
                
            elif model_name == 'Optimized_LightGBM':
                param_grid = {
                    'n_estimators': [200, 300, 400],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9]
                }
                base_model = lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1)
                
            elif model_name == 'Optimized_RandomForest':
                param_grid = {
                    'n_estimators': [200, 300, 400],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
                
            else:
                # Use original model if no specific optimization defined
                optimized_models[model_name] = models[model_name]
                continue
            
            try:
                # Use RandomizedSearchCV for efficiency
                random_search = RandomizedSearchCV(
                    base_model, 
                    param_grid, 
                    n_iter=20,
                    cv=3,
                    scoring='accuracy',
                    random_state=42,
                    n_jobs=-1
                )
                random_search.fit(X_train, y_train)
                optimized_models[model_name] = random_search.best_estimator_
                print(f"Best parameters: {random_search.best_params_}")
                print(f"Best score: {random_search.best_score_:.4f}")
                
            except Exception as e:
                print(f"Optimization failed for {model_name}: {e}")
                optimized_models[model_name] = models[model_name]
        
        # Add remaining models without optimization
        for name, model in models.items():
            if name not in optimized_models:
                optimized_models[name] = model
        
        return optimized_models
    
    def create_ensemble_models(self, base_models, X_train, y_train):
        """Create advanced ensemble models"""
        print("\n" + "=" * 60)
        print("CREATING ENSEMBLE MODELS")
        print("=" * 60)
        
        # Select best performing models for ensemble
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        model_scores = {}
        
        for name, model in base_models.items():
            try:
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                model_scores[name] = scores.mean()
            except:
                model_scores[name] = 0
        
        # Select top 4 models for ensemble
        top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:4]
        ensemble_models = {name: base_models[name] for name, _ in top_models}
        
        print(f"Selected models for ensemble: {list(ensemble_models.keys())}")
        
        # Voting Classifier
        voting_clf = VotingClassifier(
            estimators=list(ensemble_models.items()),
            voting='soft'
        )
        
        # Stacking Classifier
        stacking_clf = StackingClassifier(
            estimators=list(ensemble_models.items()),
            final_estimator=LogisticRegression(random_state=42),
            cv=3,
            n_jobs=-1
        )
        
        ensemble_models_dict = {
            'Voting_Ensemble': voting_clf,
            'Stacking_Ensemble': stacking_clf
        }
        
        # Add ensemble models to base models
        final_models = {**base_models, **ensemble_models_dict}
        
        return final_models
    
    def comprehensive_evaluation(self, models, X_train, X_test, y_train, y_test, y_labels_test):
        """Comprehensive model evaluation with detailed metrics"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)
        
        results = {}
        detailed_results = {}
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
                
                # Fit model and predict
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro')
                
                # Multi-class ROC AUC
                if y_pred_proba is not None and len(np.unique(y_test)) > 2:
                    try:
                        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
                    except:
                        roc_auc = 0
                else:
                    roc_auc = 0
                
                # Store results
                results[name] = {
                    'CV_Accuracy': cv_scores.mean(),
                    'CV_Accuracy_Std': cv_scores.std(),
                    'CV_F1': cv_f1.mean(),
                    'CV_F1_Std': cv_f1.std(),
                    'Test_Accuracy': accuracy,
                    'Test_F1': f1,
                    'ROC_AUC': roc_auc
                }
                
                # Detailed classification report
                report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_, output_dict=True)
                detailed_results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'classification_report': report,
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                
                print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                results[name] = {
                    'CV_Accuracy': 0, 'CV_Accuracy_Std': 0,
                    'CV_F1': 0, 'CV_F1_Std': 0,
                    'Test_Accuracy': 0, 'Test_F1': 0, 'ROC_AUC': 0
                }
        
        # Create results DataFrame
        results_df = pd.DataFrame(results).T.sort_values('Test_Accuracy', ascending=False)
        print(f"\n{results_df}")
        
        # Find best model
        best_model_name = results_df.index[0]
        self.best_model = detailed_results[best_model_name]['model']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Test Accuracy: {results_df.loc[best_model_name, 'Test_Accuracy']:.4f}")
        
        return results_df, detailed_results, best_model_name
    
    def create_visualizations(self, results_df, detailed_results, best_model_name, X_test, y_test, y_labels_test):
        """Create comprehensive visualizations"""
        print("\n" + "=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Model Performance Comparison
        ax1 = plt.subplot(4, 3, 1)
        top_models = results_df.head(8)
        x_pos = np.arange(len(top_models))
        plt.bar(x_pos, top_models['Test_Accuracy'], alpha=0.8, color='skyblue')
        plt.xlabel('Models')
        plt.ylabel('Test Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(x_pos, top_models.index, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # 2. Cross-validation vs Test Performance
        ax2 = plt.subplot(4, 3, 2)
        plt.scatter(top_models['CV_Accuracy'], top_models['Test_Accuracy'], alpha=0.7, s=100)
        plt.xlabel('CV Accuracy')
        plt.ylabel('Test Accuracy')
        plt.title('CV vs Test Performance')
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        plt.grid(alpha=0.3)
        
        # 3. F1 Score Comparison
        ax3 = plt.subplot(4, 3, 3)
        plt.bar(x_pos, top_models['Test_F1'], alpha=0.8, color='lightcoral')
        plt.xlabel('Models')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Comparison')
        plt.xticks(x_pos, top_models.index, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # 4. Best Model Confusion Matrix
        ax4 = plt.subplot(4, 3, 4)
        cm = detailed_results[best_model_name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 5. Classification Report Heatmap
        ax5 = plt.subplot(4, 3, 5)
        report = detailed_results[best_model_name]['classification_report']
        report_df = pd.DataFrame(report).iloc[:-1, :-1].T  # Remove support and avg rows
        sns.heatmap(report_df, annot=True, cmap='RdYlBu_r', fmt='.3f')
        plt.title(f'Classification Report - {best_model_name}')
        
        # 6. ROC AUC Comparison
        ax6 = plt.subplot(4, 3, 6)
        roc_scores = top_models['ROC_AUC']
        plt.bar(x_pos, roc_scores, alpha=0.8, color='lightgreen')
        plt.xlabel('Models')
        plt.ylabel('ROC AUC Score')
        plt.title('ROC AUC Comparison')
        plt.xticks(x_pos, top_models.index, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # 7. Model Complexity vs Performance
        ax7 = plt.subplot(4, 3, 7)
        complexity_scores = []
        for model_name in top_models.index:
            model = detailed_results[model_name]['model']
            if hasattr(model, 'n_estimators'):
                complexity_scores.append(model.n_estimators)
            elif hasattr(model, 'C'):
                complexity_scores.append(model.C * 100)  # Scale for visibility
            else:
                complexity_scores.append(100)  # Default complexity
        
        plt.scatter(complexity_scores, top_models['Test_Accuracy'], alpha=0.7, s=100)
        plt.xlabel('Model Complexity')
        plt.ylabel('Test Accuracy')
        plt.title('Complexity vs Performance')
        plt.grid(alpha=0.3)
        
        # 8. Error Analysis
        ax8 = plt.subplot(4, 3, 8)
        best_predictions = detailed_results[best_model_name]['predictions']
        error_mask = best_predictions != y_test
        
        if np.sum(error_mask) > 0:
            error_distribution = pd.Series(y_test[error_mask]).value_counts()
            error_distribution.plot(kind='bar', ax=ax8, color='red', alpha=0.7)
            plt.title('Errors by True Class')
            plt.xlabel('True Class')
            plt.ylabel('Number of Errors')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'No Errors!', ha='center', va='center', transform=ax8.transAxes)
            plt.title('Errors by True Class')
        
        # 9. Learning Curves (for tree-based models)
        ax9 = plt.subplot(4, 3, 9)
        if hasattr(detailed_results[best_model_name]['model'], 'feature_importances_'):
            importances = detailed_results[best_model_name]['model'].feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            plt.bar(range(10), importances[indices], alpha=0.8)
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
            plt.title(f'Top 10 Feature Importances - {best_model_name}')
            plt.xticks(range(10), [f'F{i}' for i in indices], rotation=45)
        else:
            plt.text(0.5, 0.5, 'Feature Importance\nNot Available', ha='center', va='center', transform=ax9.transAxes)
            plt.title('Feature Importance')
        
        # 10. Prediction Confidence Distribution
        ax10 = plt.subplot(4, 3, 10)
        if detailed_results[best_model_name]['probabilities'] is not None:
            proba = detailed_results[best_model_name]['probabilities']
            max_proba = np.max(proba, axis=1)
            plt.hist(max_proba, bins=20, alpha=0.7, color='purple')
            plt.xlabel('Prediction Confidence')
            plt.ylabel('Frequency')
            plt.title('Prediction Confidence Distribution')
            plt.axvline(np.mean(max_proba), color='red', linestyle='--', label=f'Mean: {np.mean(max_proba):.3f}')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'Probability\nNot Available', ha='center', va='center', transform=ax10.transAxes)
            plt.title('Prediction Confidence')
        
        # 11. Class Balance Analysis
        ax11 = plt.subplot(4, 3, 11)
        unique, counts = np.unique(y_test, return_counts=True)
        class_names = [self.label_encoder.classes_[i] for i in unique]
        plt.pie(counts, labels=class_names, autopct='%1.1f%%', startangle=90)
        plt.title('Test Set Class Distribution')
        
        # 12. Model Ensemble Comparison (if ensemble models exist)
        ax12 = plt.subplot(4, 3, 12)
        ensemble_models = [name for name in results_df.index if 'Ensemble' in name or 'Voting' in name or 'Stacking' in name]
        if ensemble_models:
            ensemble_scores = results_df.loc[ensemble_models, 'Test_Accuracy']
            plt.bar(range(len(ensemble_scores)), ensemble_scores, alpha=0.8, color='gold')
            plt.xlabel('Ensemble Models')
            plt.ylabel('Test Accuracy')
            plt.title('Ensemble Model Performance')
            plt.xticks(range(len(ensemble_scores)), ensemble_models, rotation=45, ha='right')
        else:
            plt.text(0.5, 0.5, 'No Ensemble\nModels', ha='center', va='center', transform=ax12.transAxes)
            plt.title('Ensemble Model Performance')
        
        plt.tight_layout()
        plt.savefig("public/blood_cancer_xgboost_results/comprehensive_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional detailed confusion matrix plot
        plt.figure(figsize=(8, 6))
        cm = detailed_results[best_model_name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Detailed Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig("public/blood_cancer_xgboost_results/confusion_matrix_detailed.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, results_df, detailed_results, best_model_name, X_test, y_test, y_labels_test):
        """Save all results and models"""
        print("\n" + "=" * 60)
        print("SAVING RESULTS")
        print("=" * 60)
        
        # Save performance results
        results_df.to_csv("public/blood_cancer_xgboost_results/model_performance.csv")
        
        # Save best model
        joblib.dump(self.best_model, "public/blood_cancer_xgboost_results/best_model.joblib")
        joblib.dump(self.scaler, "public/blood_cancer_xgboost_results/scaler.joblib")
        joblib.dump(self.label_encoder, "public/blood_cancer_xgboost_results/label_encoder.joblib")
        
        if self.feature_selector:
            joblib.dump(self.feature_selector, "public/blood_cancer_xgboost_results/feature_selector.joblib")
        
        if self.sampler:
            joblib.dump(self.sampler, "public/blood_cancer_xgboost_results/sampler.joblib")
        
        # Save detailed results
        best_results = detailed_results[best_model_name]
        
        # Classification report
        with open("public/blood_cancer_xgboost_results/classification_report.json", 'w') as f:
            json.dump(best_results['classification_report'], f, indent=2)
        
        # Confusion matrix
        np.savetxt("public/blood_cancer_xgboost_results/confusion_matrix.csv", 
                   best_results['confusion_matrix'], delimiter=",", fmt="%d")
        
        # Predictions
        predictions_df = pd.DataFrame({
            'True_Label': y_labels_test,
            'Predicted_Label': [self.label_encoder.classes_[pred] for pred in best_results['predictions']],
            'True_Index': y_test,
            'Predicted_Index': best_results['predictions']
        })
        
        if best_results['probabilities'] is not None:
            for i, class_name in enumerate(self.label_encoder.classes_):
                predictions_df[f'Probability_{class_name}'] = best_results['probabilities'][:, i]
        
        predictions_df.to_csv("public/blood_cancer_xgboost_results/predictions.csv", index=False)
        
        # Model summary
        summary = {
            'best_model': best_model_name,
            'test_accuracy': float(results_df.loc[best_model_name, 'Test_Accuracy']),
            'test_f1': float(results_df.loc[best_model_name, 'Test_F1']),
            'roc_auc': float(results_df.loc[best_model_name, 'ROC_AUC']),
            'cv_accuracy': float(results_df.loc[best_model_name, 'CV_Accuracy']),
            'cv_f1': float(results_df.loc[best_model_name, 'CV_F1']),
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'sample_count': len(X_test),
            'class_names': self.label_encoder.classes_.tolist(),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open("public/blood_cancer_xgboost_results/model_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("All results saved successfully!")
        print(f"Best Model: {best_model_name}")
        print(f"Test Accuracy: {summary['test_accuracy']:.4f}")
        print(f"Test F1 Score: {summary['test_f1']:.4f}")
        print(f"ROC AUC: {summary['roc_auc']:.4f}")
        
        return summary
    
    def predict_new_sample(self, sample_data):
        """Predict cancer type for new sample"""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        # Process the sample through the same pipeline
        sample_df = pd.DataFrame([sample_data])
        
        # Apply the same preprocessing steps
        # Note: In a production environment, you'd want to save and load the preprocessing pipeline
        sample_processed = self.scaler.transform(sample_df)
        
        # Make prediction
        prediction = self.best_model.predict(sample_processed)[0]
        prediction_proba = self.best_model.predict_proba(sample_processed)[0]
        
        # Convert back to original labels
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        
        result = {
            'predicted_class': predicted_class,
            'confidence': float(np.max(prediction_proba)),
            'all_probabilities': {
                self.label_encoder.classes_[i]: float(prob) 
                for i, prob in enumerate(prediction_proba)
            }
        }
        
        return result
    
    def run_complete_analysis(self, file_path):
        """Run the complete analysis pipeline"""
        print("🔬 ADVANCED BLOOD CANCER CLASSIFICATION ANALYSIS")
        print("=" * 80)
        
        # Load and explore data
        df = self.load_and_explore_data(file_path)
        
        # Advanced preprocessing
        X_processed, y_encoded, y_original = self.advanced_preprocessing(df)
        
        # Feature selection
        X_selected, selected_features = self.advanced_feature_selection(X_processed, y_encoded)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train)
        
        # Create models
        models = self.create_advanced_models()
        
        # Optimize hyperparameters
        optimized_models = self.hyperparameter_optimization(models, X_train_balanced, y_train_balanced)
        
        # Create ensemble models
        final_models = self.create_ensemble_models(optimized_models, X_train_balanced, y_train_balanced)
        
        # Comprehensive evaluation
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        results_df, detailed_results, best_model_name = self.comprehensive_evaluation(
            final_models, X_train_balanced, X_test, y_train_balanced, y_test, y_test_labels
        )
        
        # Create visualizations
        self.create_visualizations(results_df, detailed_results, best_model_name, X_test, y_test, y_test_labels)
        
        # Save results
        summary = self.save_results(results_df, detailed_results, best_model_name, X_test, y_test, y_test_labels)
        
        print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("📁 All results saved in: public/blood_cancer_xgboost_results/")
        
        return summary, self.best_model

# Main execution function
def main():
    """Main function to run the analysis"""
    
    # Initialize the classifier
    classifier = EnhancedBloodCancerClassifier()
    
    # Specify your dataset path
    # Replace with your actual dataset path
    file_path = "public/blood_cancer_diseases_dataset.csv"
    
    try:
        # Run complete analysis
        summary, best_model = classifier.run_complete_analysis(file_path)
        
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print(f"✅ Best Model: {summary['best_model']}")
        print(f"✅ Test Accuracy: {summary['test_accuracy']:.4f}")
        print(f"✅ Test F1 Score: {summary['test_f1']:.4f}")
        print(f"✅ ROC AUC Score: {summary['roc_auc']:.4f}")
        print(f"✅ Features Used: {summary['feature_count']}")
        print(f"✅ Test Samples: {summary['sample_count']}")
        print(f"✅ Classes: {', '.join(summary['class_names'])}")
        
        # Example of making a prediction on new data
        print("\n" + "="*80)
        print("EXAMPLE PREDICTION")
        print("="*80)
        print("To make predictions on new samples, use:")
        print("result = classifier.predict_new_sample(new_sample_data)")
        print("print(result)")
        
        return classifier, summary
        
    except FileNotFoundError:
        print(f"❌ Error: Dataset file not found at {file_path}")
        print("Please update the file_path variable with the correct path to your dataset.")
        return None, None
    
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # Run the main analysis
    classifier, summary = main()
    
    # Additional utility functions
    def load_saved_model():
        """Load a previously saved model"""
        try:
            best_model = joblib.load("public/blood_cancer_xgboost_results/best_model.joblib")
            scaler = joblib.load("public/blood_cancer_xgboost_results/scaler.joblib")
            label_encoder = joblib.load("public/blood_cancer_xgboost_results/label_encoder.joblib")
            
            print("✅ Models loaded successfully!")
            return best_model, scaler, label_encoder
        except FileNotFoundError:
            print("❌ No saved models found. Please run the training first.")
            return None, None, None
    
    def make_prediction_from_saved_model(sample_data):
        """Make prediction using saved model"""
        model, scaler, label_encoder = load_saved_model()
        if model is None:
            return None
        
        # Process sample
        sample_df = pd.DataFrame([sample_data])
        sample_scaled = scaler.transform(sample_df)
        
        # Predict
        prediction = model.predict(sample_scaled)[0]
        prediction_proba = model.predict_proba(sample_scaled)[0]
        
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(np.max(prediction_proba)),
            'all_probabilities': {
                label_encoder.classes_[i]: float(prob) 
                for i, prob in enumerate(prediction_proba)
            }
        }