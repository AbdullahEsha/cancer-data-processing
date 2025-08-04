# Enhanced Blood Cancer Classification System with PCA and Advanced Feature Engineering
# This comprehensive system includes PCA, advanced feature engineering, and ensemble methods

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Advanced ML libraries
from sklearn.model_selection import (train_test_split, GridSearchCV, StratifiedKFold, 
                                   cross_val_score, validation_curve, learning_curve)
from sklearn.preprocessing import (LabelEncoder, StandardScaler, MinMaxScaler, 
                                 RobustScaler, PolynomialFeatures, PowerTransformer)
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.feature_selection import (SelectKBest, f_classif, chi2, mutual_info_classif,
                                     RFE, SelectFromModel, VarianceThreshold)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

# Metrics and evaluation
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                           roc_auc_score, precision_recall_fscore_support, roc_curve, 
                           auc, f1_score, precision_score, recall_score)

# Models
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier,
                            BaggingClassifier, StackingClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LassoCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Additional libraries
from scipy import stats
from scipy.stats import boxcox, yeojohnson
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import combinations
import time

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced feature engineering transformer"""
    
    def __init__(self, poly_degree=2, include_interaction=True, include_statistical=True, 
                 include_clustering=True, n_clusters=5):
        self.poly_degree = poly_degree
        self.include_interaction = include_interaction
        self.include_statistical = include_statistical
        self.include_clustering = include_clustering
        self.n_clusters = n_clusters
        self.poly_features = None
        self.kmeans = None
        self.feature_names = None
        
    def fit(self, X, y=None):
        # Identify numeric columns
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.include_clustering and len(self.numeric_columns) > 1:
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            self.kmeans.fit(X[self.numeric_columns])
        
        return self
    
    def transform(self, X):
        X_new = X.copy()
        
        # Statistical features
        if self.include_statistical and len(self.numeric_columns) > 0:
            for col in self.numeric_columns:
                # Z-scores
                X_new[f'{col}_zscore'] = (X[col] - X[col].mean()) / X[col].std()
                
                # Percentile ranks
                X_new[f'{col}_percentile'] = X[col].rank(pct=True)
                
                # Log transformation (for positive values)
                if (X[col] > 0).all():
                    X_new[f'{col}_log'] = np.log1p(X[col])
                
                # Square root transformation
                if (X[col] >= 0).all():
                    X_new[f'{col}_sqrt'] = np.sqrt(X[col])
        
        # Interaction features for numeric columns
        if self.include_interaction and len(self.numeric_columns) > 1:
            for col1, col2 in combinations(self.numeric_columns[:5], 2):  # Limit to first 5 to avoid explosion
                X_new[f'{col1}_{col2}_ratio'] = X[col1] / (X[col2] + 1e-8)
                X_new[f'{col1}_{col2}_product'] = X[col1] * X[col2]
                X_new[f'{col1}_{col2}_diff'] = X[col1] - X[col2]
        
        # Clustering features
        if self.include_clustering and self.kmeans is not None:
            clusters = self.kmeans.predict(X[self.numeric_columns])
            X_new['cluster'] = clusters
            
            # Distance to cluster centers
            distances = self.kmeans.transform(X[self.numeric_columns])
            X_new['min_cluster_distance'] = distances.min(axis=1)
            X_new['cluster_distance_std'] = distances.std(axis=1)
        
        return X_new

class EnhancedBloodCancerClassifier:
    def __init__(self, csv_path=None):
        self.csv_path = csv_path
        self.df_raw = None
        self.df_clean = None
        self.df_enhanced = None
        self.X_encoded = None
        self.y_encoded = None
        self.label_encoder = None
        self.scalers = {}
        self.pca_models = {}
        self.feature_selectors = {}
        self.models = {}
        self.results = {}
        self.pca_results = {}
        self.feature_importance = {}
        
    def load_data(self):
        """Load and perform initial inspection of the dataset"""
        if self.csv_path:
            try:
                self.df_raw = pd.read_csv(self.csv_path)
                print(f"Dataset loaded successfully from {self.csv_path}")
            except FileNotFoundError:
                print(f"File {self.csv_path} not found. Creating sample dataset...")
                self.create_sample_data()
        else:
            print("No CSV path provided. Creating sample dataset...")
            self.create_sample_data()
            
        print(f"Dataset shape: {self.df_raw.shape}")
        print(f"Columns: {list(self.df_raw.columns)}")
        return self.df_raw.head()
    
    def create_sample_data(self):
        """Create a realistic sample dataset for demonstration"""
        print("Creating realistic blood cancer dataset...")
        np.random.seed(42)
        n_samples = 2000

        # Define cancer type profiles with realistic characteristics
        cancer_profiles = {
            'AML': {
                'age_range': (40, 80), 'wbc_factor': 3.0, 'platelet_factor': 0.5,
                'genetic_probs': {'FLT3': 0.3, 'TP53': 0.2, 'Normal': 0.3, 'BCR-ABL': 0.1, 'MYC': 0.1},
                'bma_pos_prob': 0.85, 'lnb_pos_prob': 0.4, 'spep_abnormal_prob': 0.3,
                'diagnosis_probs': {'Confirmed': 0.7, 'Suspected': 0.2, 'Ruled Out': 0.1}
            },
            'ALL': {
                'age_range': (20, 60), 'wbc_factor': 4.0, 'platelet_factor': 0.4,
                'genetic_probs': {'Normal': 0.4, 'TP53': 0.15, 'MYC': 0.25, 'FLT3': 0.1, 'BCR-ABL': 0.1},
                'bma_pos_prob': 0.90, 'lnb_pos_prob': 0.6, 'spep_abnormal_prob': 0.2,
                'diagnosis_probs': {'Confirmed': 0.75, 'Suspected': 0.15, 'Ruled Out': 0.1}
            },
            'CLL': {
                'age_range': (55, 85), 'wbc_factor': 2.5, 'platelet_factor': 0.7,
                'genetic_probs': {'Normal': 0.5, 'TP53': 0.25, 'FLT3': 0.1, 'BCR-ABL': 0.05, 'MYC': 0.1},
                'bma_pos_prob': 0.75, 'lnb_pos_prob': 0.8, 'spep_abnormal_prob': 0.4,
                'diagnosis_probs': {'Confirmed': 0.6, 'Suspected': 0.3, 'Ruled Out': 0.1}
            },
            'CML': {
                'age_range': (30, 70), 'wbc_factor': 5.0, 'platelet_factor': 1.2,
                'genetic_probs': {'BCR-ABL': 0.95, 'Normal': 0.03, 'FLT3': 0.01, 'TP53': 0.005, 'MYC': 0.005},
                'bma_pos_prob': 0.95, 'lnb_pos_prob': 0.3, 'spep_abnormal_prob': 0.1,
                'diagnosis_probs': {'Confirmed': 0.8, 'Suspected': 0.15, 'Ruled Out': 0.05}
            },
            'Lymphoma': {
                'age_range': (25, 75), 'wbc_factor': 1.5, 'platelet_factor': 0.8,
                'genetic_probs': {'MYC': 0.3, 'Normal': 0.4, 'TP53': 0.15, 'FLT3': 0.1, 'BCR-ABL': 0.05},
                'bma_pos_prob': 0.6, 'lnb_pos_prob': 0.9, 'spep_abnormal_prob': 0.3,
                'diagnosis_probs': {'Confirmed': 0.65, 'Suspected': 0.25, 'Ruled Out': 0.1}
            },
            'Multiple Myeloma': {
                'age_range': (50, 85), 'wbc_factor': 0.8, 'platelet_factor': 0.6,
                'genetic_probs': {'Normal': 0.3, 'TP53': 0.3, 'MYC': 0.2, 'FLT3': 0.1, 'BCR-ABL': 0.1},
                'bma_pos_prob': 0.8, 'lnb_pos_prob': 0.2, 'spep_abnormal_prob': 0.95,
                'diagnosis_probs': {'Confirmed': 0.7, 'Suspected': 0.2, 'Ruled Out': 0.1}
            }
        }

        data_list = []
        samples_per_type = n_samples // len(cancer_profiles)

        for cancer_type, profile in cancer_profiles.items():
            for _ in range(samples_per_type):
                # Generate patient data
                age = np.random.randint(profile['age_range'][0], profile['age_range'][1])
                
                # WBC count - log-normal distribution for realistic values
                base_wbc = 8000 + (age - 40) * 200
                wbc_count = np.random.exponential(base_wbc * profile['wbc_factor'])
                wbc_count = max(1000, min(500000, wbc_count))  # Realistic bounds
                
                # Platelet count
                base_platelet = 250000
                platelet_count = max(10000, np.random.normal(base_platelet * profile['platelet_factor'], 50000))
                
                # Additional lab values for more features
                hemoglobin = max(5.0, np.random.normal(12.0, 2.0))
                ldh = max(100, np.random.normal(200 + age * 2, 50))
                
                # Categorical variables
                genetic_data = np.random.choice(list(profile['genetic_probs'].keys()), 
                                              p=list(profile['genetic_probs'].values()))
                
                bma_result = 'Positive' if np.random.random() < profile['bma_pos_prob'] else 'Negative'
                lnb_result = 'Positive' if np.random.random() < profile['lnb_pos_prob'] else 'Negative'
                spep_result = 'Abnormal' if np.random.random() < profile['spep_abnormal_prob'] else 'Normal'
                
                # Add lumbar puncture result
                lp_probs = [0.3, 0.5, 0.2]  # Positive, Negative, Not Done
                lp_result = np.random.choice(['Positive', 'Negative', 'Not Done'], p=lp_probs)
                
                # Add diagnosis result
                diagnosis_result = np.random.choice(list(profile['diagnosis_probs'].keys()),
                                                  p=list(profile['diagnosis_probs'].values()))

                data_list.append({
                    'Age': age,
                    'Cancer_Type(AML, ALL, CLL)': cancer_type,
                    'Total WBC count(/cumm)': int(wbc_count),
                    'Platelet Count(/cumm)': int(platelet_count),
                    'Hemoglobin(g/dL)': round(hemoglobin, 1),
                    'LDH(U/L)': int(ldh),
                    'Genetic_Data(BCR-ABL, FLT3)': genetic_data,
                    'Bone Marrow Aspiration(Positive / Negative / Not Done)': bma_result,
                    'Lymph Node Biopsy(Positive / Negative / Not Done)': lnb_result,
                    'Serum Protein Electrophoresis (SPEP)(Normal / Abnormal)': spep_result,
                    'Lumbar Puncture (Spinal Tap)': lp_result,
                    'Diagnosis_Result': diagnosis_result
                })

        self.df_raw = pd.DataFrame(data_list)
        # Shuffle the data
        self.df_raw = self.df_raw.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Sample dataset created with {len(self.df_raw)} samples")

    def clean_data(self):
        """Clean and preprocess the dataset"""
        print("\n" + "="*60)
        print("DATA CLEANING AND PREPROCESSING")
        print("="*60)
        
        df = self.df_raw.copy()
        
        # Clean column names
        df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)
        df.columns = df.columns.str.replace(' ', '_')
        df.columns = df.columns.str.lower()
        
        # Define column mapping for standardization
        column_mapping = {
            'age': 'age',
            'cancer_typeaml_all_cll': 'cancer_type',
            'total_wbc_countcumm': 'total_wbc_count',
            'platelet_countcumm': 'platelet_count',
            'hemoglobingdl': 'hemoglobin',
            'ldhul': 'ldh',
            'genetic_databcrabl_flt3': 'genetic_data',
            'bone_marrow_aspirationpositive__negative__not_done': 'bone_marrow_aspiration',
            'lymph_node_biopsypositive__negative__not_done': 'lymph_node_biopsy',
            'serum_protein_electrophoresis_spepnormal__abnormal': 'spep_result',
            'lumbar_puncture_spinal_tap': 'lumbar_puncture',
            'diagnosis_result': 'diagnosis_result'
        }
        
        # Apply column renaming
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
                print(f"Renamed '{old_name}' to '{new_name}'")
        
        print(f"\nColumns after renaming: {list(df.columns)}")
        
        # Convert numeric columns
        numeric_columns = ['age', 'total_wbc_count', 'platelet_count', 'hemoglobin', 'ldh']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"Converted '{col}' to numeric")
        
        # Remove rows with NaN values in critical columns
        critical_columns = [col for col in numeric_columns if col in df.columns]
        if critical_columns:
            df = df.dropna(subset=critical_columns)
        
        self.df_clean = df
        print(f"\nFinal cleaned dataset shape: {self.df_clean.shape}")
        print(f"Columns: {list(self.df_clean.columns)}")
        
        return self.df_clean.head()

    def advanced_feature_engineering(self):
        """Perform advanced feature engineering"""
        print("\n" + "="*60)
        print("ADVANCED FEATURE ENGINEERING")
        print("="*60)
        
        if self.df_clean is None:
            print("Please run clean_data() first")
            return
        
        df_enhanced = self.df_clean.copy()
        
        # Medical risk scoring
        risk_score = 0
        
        # Age-based risk
        if 'age' in df_enhanced.columns:
            df_enhanced['age_risk_young'] = (df_enhanced['age'] < 25).astype(int)
            df_enhanced['age_risk_elderly'] = (df_enhanced['age'] > 70).astype(int)
            df_enhanced['age_squared'] = df_enhanced['age'] ** 2
            df_enhanced['age_normalized'] = (df_enhanced['age'] - df_enhanced['age'].mean()) / df_enhanced['age'].std()
            risk_score += (df_enhanced['age'] > 65).astype(int) * 2
        
        # Laboratory value engineering
        if 'total_wbc_count' in df_enhanced.columns:
            # WBC risk categories
            df_enhanced['wbc_very_high'] = (df_enhanced['total_wbc_count'] > 100000).astype(int)
            df_enhanced['wbc_high'] = ((df_enhanced['total_wbc_count'] > 30000) & 
                                     (df_enhanced['total_wbc_count'] <= 100000)).astype(int)
            df_enhanced['wbc_normal'] = ((df_enhanced['total_wbc_count'] >= 4000) & 
                                       (df_enhanced['total_wbc_count'] <= 30000)).astype(int)
            df_enhanced['wbc_low'] = (df_enhanced['total_wbc_count'] < 4000).astype(int)
            
            # Log transformation for WBC
            df_enhanced['log_wbc'] = np.log1p(df_enhanced['total_wbc_count'])
            
            # WBC percentiles
            df_enhanced['wbc_percentile'] = df_enhanced['total_wbc_count'].rank(pct=True)
            
            risk_score += (df_enhanced['total_wbc_count'] > 50000).astype(int) * 3
        
        if 'platelet_count' in df_enhanced.columns:
            # Platelet risk categories
            df_enhanced['platelet_very_low'] = (df_enhanced['platelet_count'] < 50000).astype(int)
            df_enhanced['platelet_low'] = ((df_enhanced['platelet_count'] >= 50000) & 
                                         (df_enhanced['platelet_count'] < 150000)).astype(int)
            df_enhanced['platelet_normal'] = ((df_enhanced['platelet_count'] >= 150000) & 
                                            (df_enhanced['platelet_count'] <= 450000)).astype(int)
            df_enhanced['platelet_high'] = (df_enhanced['platelet_count'] > 450000).astype(int)
            
            # Log transformation
            df_enhanced['log_platelet'] = np.log1p(df_enhanced['platelet_count'])
            
            risk_score += (df_enhanced['platelet_count'] < 100000).astype(int) * 2
        
        if 'hemoglobin' in df_enhanced.columns:
            # Anemia indicators
            df_enhanced['severe_anemia'] = (df_enhanced['hemoglobin'] < 8.0).astype(int)
            df_enhanced['moderate_anemia'] = ((df_enhanced['hemoglobin'] >= 8.0) & 
                                            (df_enhanced['hemoglobin'] < 11.0)).astype(int)
            df_enhanced['mild_anemia'] = ((df_enhanced['hemoglobin'] >= 11.0) & 
                                        (df_enhanced['hemoglobin'] < 12.0)).astype(int)
            df_enhanced['normal_hb'] = (df_enhanced['hemoglobin'] >= 12.0).astype(int)
            
            risk_score += (df_enhanced['hemoglobin'] < 10.0).astype(int) * 2
        
        if 'ldh' in df_enhanced.columns:
            # LDH risk levels
            df_enhanced['ldh_very_high'] = (df_enhanced['ldh'] > 500).astype(int)
            df_enhanced['ldh_high'] = ((df_enhanced['ldh'] > 300) & (df_enhanced['ldh'] <= 500)).astype(int)
            df_enhanced['ldh_normal'] = (df_enhanced['ldh'] <= 300).astype(int)
            
            # Log transformation
            df_enhanced['log_ldh'] = np.log1p(df_enhanced['ldh'])
            
            risk_score += (df_enhanced['ldh'] > 400).astype(int) * 1
        
        # Genetic risk factors
        if 'genetic_data' in df_enhanced.columns:
            high_risk_genetics = ['FLT3', 'TP53', 'BCR-ABL']
            df_enhanced['genetic_high_risk'] = df_enhanced['genetic_data'].isin(high_risk_genetics).astype(int)
            risk_score += df_enhanced['genetic_high_risk'] * 3
        
        # Clinical test combinations
        if 'bone_marrow_aspiration' in df_enhanced.columns:
            df_enhanced['bma_positive'] = (df_enhanced['bone_marrow_aspiration'] == 'Positive').astype(int)
            risk_score += df_enhanced['bma_positive'] * 2
        
        if 'lymph_node_biopsy' in df_enhanced.columns:
            df_enhanced['lnb_positive'] = (df_enhanced['lymph_node_biopsy'] == 'Positive').astype(int)
            risk_score += df_enhanced['lnb_positive'] * 1
        
        if 'spep_result' in df_enhanced.columns:
            df_enhanced['spep_abnormal'] = (df_enhanced['spep_result'] == 'Abnormal').astype(int)
            risk_score += df_enhanced['spep_abnormal'] * 1
        
        # Composite risk score
        df_enhanced['total_risk_score'] = risk_score
        df_enhanced['risk_category'] = pd.cut(risk_score, 
                                            bins=[-1, 2, 5, 8, 20], 
                                            labels=['Low', 'Moderate', 'High', 'Very High'])
        
        # Interaction features
        numeric_cols = ['age', 'total_wbc_count', 'platelet_count', 'hemoglobin', 'ldh']
        available_numeric = [col for col in numeric_cols if col in df_enhanced.columns]
        
        if len(available_numeric) >= 2:
            # Ratios
            if 'total_wbc_count' in available_numeric and 'age' in available_numeric:
                df_enhanced['wbc_age_ratio'] = df_enhanced['total_wbc_count'] / (df_enhanced['age'] + 1)
            
            if 'platelet_count' in available_numeric and 'age' in available_numeric:
                df_enhanced['platelet_age_ratio'] = df_enhanced['platelet_count'] / (df_enhanced['age'] + 1)
            
            if 'total_wbc_count' in available_numeric and 'platelet_count' in available_numeric:
                df_enhanced['wbc_platelet_ratio'] = (df_enhanced['total_wbc_count'] / 
                                                   (df_enhanced['platelet_count'] + 1))
            
            if 'hemoglobin' in available_numeric and 'age' in available_numeric:
                df_enhanced['hb_age_interaction'] = df_enhanced['hemoglobin'] * df_enhanced['age']
        
        # Clustering features for pattern recognition
        numeric_features_for_clustering = [col for col in available_numeric if col in df_enhanced.columns]
        if len(numeric_features_for_clustering) >= 2:
            # Standardize features for clustering
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df_enhanced[numeric_features_for_clustering])
            
            # K-means clustering
            for n_clusters in [3, 5, 7]:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                df_enhanced[f'cluster_{n_clusters}'] = kmeans.fit_predict(scaled_features)
        
        self.df_enhanced = df_enhanced
        print(f"Enhanced dataset shape: {self.df_enhanced.shape}")
        print(f"New features added: {len(self.df_enhanced.columns) - len(self.df_clean.columns)}")
        
        return self.df_enhanced.head()

    def prepare_ml_features(self):
        """Prepare features for machine learning with encoding"""
        print("\n" + "="*60)
        print("PREPARING FEATURES FOR MACHINE LEARNING")
        print("="*60)
        
        if not hasattr(self, 'df_enhanced'):
            print("Please run advanced_feature_engineering() first")
            return
        
        # Define target column
        target_col = 'cancer_type'
        if target_col not in self.df_enhanced.columns:
            print(f"Target column '{target_col}' not found!")
            return
        
        # Select feature columns (exclude target and irrelevant columns)
        exclude_cols = [target_col, 'diagnosis_result']
        feature_columns = [col for col in self.df_enhanced.columns if col not in exclude_cols]
        
        X = self.df_enhanced[feature_columns].copy()
        y = self.df_enhanced[target_col].copy()
        
        # Handle categorical variables with one-hot encoding
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"Categorical features: {categorical_features}")
        
        # One-hot encode categorical variables
        if categorical_features:
            X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)
        else:
            X_encoded = X.copy()
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.X_encoded = X_encoded
        self.y_encoded = y_encoded
        self.y = y
        
        print(f"Features shape: {X_encoded.shape}")
        print(f"Target classes: {self.label_encoder.classes_}")
        print(f"Class distribution:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            count = np.sum(y_encoded == i)
            print(f"  {class_name}: {count} ({count/len(y_encoded)*100:.1f}%)")
        
        return X_encoded.head()

    def perform_pca_analysis(self, n_components_range=(2, 50)):
        """Perform comprehensive PCA analysis"""
        print("\n" + "="*60)
        print("PRINCIPAL COMPONENT ANALYSIS")
        print("="*60)
        
        if self.X_encoded is None:
            print("Please run prepare_ml_features() first")
            return
        
        # Standardize features for PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_encoded)
        self.scalers['standard'] = scaler
        
        # PCA with different numbers of components
        explained_variance_ratios = []
        cumulative_variance_ratios = []
        n_components_list = range(n_components_range[0], 
                                min(n_components_range[1], X_scaled.shape[1]))
        
        for n_comp in n_components_list:
            pca = PCA(n_components=n_comp, random_state=42)
            pca.fit(X_scaled)
            explained_variance_ratios.append(pca.explained_variance_ratio_)
            cumulative_variance_ratios.append(pca.explained_variance_ratio_.sum())
        
        # Find optimal number of components (95% variance explained)
        optimal_components = None
        for i, cum_var in enumerate(cumulative_variance_ratios):
            if cum_var >= 0.95:
                optimal_components = n_components_list[i]
                break
        
        if optimal_components is None:
            optimal_components = n_components_list[-1]
        
        print(f"Optimal number of components (95% variance): {optimal_components}")
        
        # Fit final PCA models
        pca_models = {}
        
        # Standard PCA
        pca_standard = PCA(n_components=optimal_components, random_state=42)
        X_pca_standard = pca_standard.fit_transform(X_scaled)
        pca_models['standard'] = {
            'model': pca_standard,
            'data': X_pca_standard,
            'explained_variance_ratio': pca_standard.explained_variance_ratio_,
            'cumulative_variance': pca_standard.explained_variance_ratio_.cumsum()
        }
        
        # Kernel PCA (RBF)
        try:
            pca_kernel = KernelPCA(n_components=min(optimal_components, 20), 
                                 kernel='rbf', random_state=42)
            X_pca_kernel = pca_kernel.fit_transform(X_scaled)
            pca_models['kernel_rbf'] = {
                'model': pca_kernel,
                'data': X_pca_kernel
            }
        except Exception as e:
            print(f"Kernel PCA failed: {e}")
        
        # Store PCA results
        self.pca_models = pca_models
        self.pca_results = {
            'optimal_components': optimal_components,
            'explained_variance_ratios': explained_variance_ratios,
            'cumulative_variance_ratios': cumulative_variance_ratios,
            'n_components_list': list(n_components_list)
        }
        
        # Create PCA visualizations
        self.plot_pca_analysis()
        
        return pca_models

    def plot_pca_analysis(self):
        """Create comprehensive PCA visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Explained variance ratio
        if 'explained_variance_ratios' in self.pca_results:
            n_comp_list = self.pca_results['n_components_list']
            cum_var_ratios = self.pca_results['cumulative_variance_ratios']
            
            axes[0, 0].plot(n_comp_list, cum_var_ratios, 'bo-', markersize=4)
            axes[0, 0].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
            axes[0, 0].axvline(x=self.pca_results['optimal_components'], 
                             color='g', linestyle='--', label=f"Optimal: {self.pca_results['optimal_components']}")
            axes[0, 0].set_xlabel('Number of Components')
            axes[0, 0].set_ylabel('Cumulative Explained Variance Ratio')
            axes[0, 0].set_title('PCA: Cumulative Explained Variance')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Individual component variance
        if 'standard' in self.pca_models:
            pca_std = self.pca_models['standard']
            component_variance = pca_std['explained_variance_ratio'][:20]  # Show first 20
            axes[0, 1].bar(range(1, len(component_variance) + 1), component_variance)
            axes[0, 1].set_xlabel('Principal Component')
            axes[0, 1].set_ylabel('Explained Variance Ratio')
            axes[0, 1].set_title('Individual Component Variance')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 2D PCA scatter plot
        if 'standard' in self.pca_models:
            X_pca = self.pca_models['standard']['data']
            scatter = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                       c=self.y_encoded, cmap='viridis', alpha=0.6)
            axes[1, 0].set_xlabel('First Principal Component')
            axes[1, 0].set_ylabel('Second Principal Component')
            axes[1, 0].set_title('PCA: First Two Components')
            plt.colorbar(scatter, ax=axes[1, 0])
        
        # 4. Feature importance in first few components
        if 'standard' in self.pca_models:
            pca_components = self.pca_models['standard']['model'].components_
            feature_importance = np.abs(pca_components[:3]).mean(axis=0)  # Average of first 3 components
            feature_names = self.X_encoded.columns[:20]  # Show top 20 features
            
            if len(feature_names) > len(feature_importance):
                feature_names = feature_names[:len(feature_importance)]
            elif len(feature_importance) > len(feature_names):
                feature_importance = feature_importance[:len(feature_names)]
            
            sorted_idx = np.argsort(feature_importance)[-20:]  # Top 20
            axes[1, 1].barh(range(len(sorted_idx)), feature_importance[sorted_idx])
            axes[1, 1].set_yticks(range(len(sorted_idx)))
            axes[1, 1].set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
            axes[1, 1].set_xlabel('Average Absolute Component Loading')
            axes[1, 1].set_title('Feature Importance in PCA')
        
        plt.tight_layout()
        plt.show()

    def apply_feature_selection(self):
        """Apply various feature selection techniques"""
        print("\n" + "="*60)
        print("FEATURE SELECTION")
        print("="*60)
        
        if self.X_encoded is None:
            print("Please run prepare_ml_features() first")
            return
        
        X_scaled = StandardScaler().fit_transform(self.X_encoded)
        feature_selectors = {}
        
        # 1. Variance Threshold
        variance_selector = VarianceThreshold(threshold=0.01)
        variance_selector.fit(X_scaled)
        feature_selectors['variance'] = variance_selector
        
        # 2. SelectKBest with different scoring functions
        k_best = min(50, X_scaled.shape[1] - 1)
        
        # F-test
        f_selector = SelectKBest(score_func=f_classif, k=k_best)
        f_selector.fit(X_scaled, self.y_encoded)
        feature_selectors['f_classif'] = f_selector
        
        # Mutual Information
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
        mi_selector.fit(X_scaled, self.y_encoded)
        feature_selectors['mutual_info'] = mi_selector
        
        # 3. RFE with Random Forest
        rf_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        rfe_selector = RFE(estimator=rf_estimator, n_features_to_select=k_best)
        rfe_selector.fit(X_scaled, self.y_encoded)
        feature_selectors['rfe'] = rfe_selector
        
        # 4. LASSO-based selection
        lasso_cv = LassoCV(cv=5, random_state=42)
        lasso_selector = SelectFromModel(lasso_cv)
        lasso_selector.fit(X_scaled, self.y_encoded)
        feature_selectors['lasso'] = lasso_selector
        
        self.feature_selectors = feature_selectors
        
        # Print selection results
        for name, selector in feature_selectors.items():
            if hasattr(selector, 'get_support'):
                n_selected = selector.get_support().sum()
                print(f"{name}: Selected {n_selected} features out of {X_scaled.shape[1]}")
        
        return feature_selectors

    def setup_enhanced_models(self):
        """Setup enhanced models with PCA and feature selection pipelines"""
        print("\n" + "="*60)
        print("SETTING UP ENHANCED MODEL PIPELINES")
        print("="*60)
        
        models_config = {}
        
        # Base models
        base_models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Extra Trees': ExtraTreesClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'MLP Neural Network': MLPClassifier(random_state=42, max_iter=1000),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        }
        
        if XGBOOST_AVAILABLE:
            base_models['XGBoost'] = XGBClassifier(random_state=42, eval_metric='mlogloss')
        
        if LIGHTGBM_AVAILABLE:
            base_models['LightGBM'] = LGBMClassifier(random_state=42)
        
        # Create pipelines with different preprocessing strategies
        preprocessing_strategies = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
        }
        
        # Models without PCA
        for model_name, model in base_models.items():
            for prep_name, scaler in preprocessing_strategies.items():
                if model_name in ['Random Forest', 'Extra Trees', 'Gradient Boosting', 'Decision Tree']:
                    # Tree-based models don't need scaling
                    if prep_name == 'standard':
                        models_config[f'{model_name}_NoScaling'] = {
                            'pipeline': Pipeline([('model', model)]),
                            'params': self._get_model_params(model_name, prefix='model__')
                        }
                else:
                    models_config[f'{model_name}_{prep_name}'] = {
                        'pipeline': Pipeline([('scaler', scaler), ('model', model)]),
                        'params': self._get_model_params(model_name, prefix='model__')
                    }
        
        # Models with PCA
        if hasattr(self, 'pca_models') and 'standard' in self.pca_models:
            optimal_components = self.pca_results['optimal_components']
            
            for model_name, model in base_models.items():
                # PCA + Model pipeline
                models_config[f'{model_name}_PCA'] = {
                    'pipeline': Pipeline([
                        ('scaler', StandardScaler()),
                        ('pca', PCA(n_components=optimal_components, random_state=42)),
                        ('model', model)
                    ]),
                    'params': self._get_model_params(model_name, prefix='model__')
                }
        
        # Ensemble models
        if len(base_models) >= 3:
            # Voting Classifier
            voting_estimators = [(name, model) for name, model in list(base_models.items())[:5]]
            voting_clf = VotingClassifier(estimators=voting_estimators, voting='soft')
            models_config['Voting_Classifier'] = {
                'pipeline': Pipeline([('scaler', StandardScaler()), ('model', voting_clf)]),
                'params': {}
            }
            
            # Stacking Classifier
            stacking_estimators = voting_estimators
            stacking_clf = StackingClassifier(
                estimators=stacking_estimators,
                final_estimator=LogisticRegression(random_state=42),
                cv=3
            )
            models_config['Stacking_Classifier'] = {
                'pipeline': Pipeline([('scaler', StandardScaler()), ('model', stacking_clf)]),
                'params': {}
            }
        
        self.models = models_config
        print(f"Created {len(models_config)} enhanced model configurations")
        return models_config

    def _get_model_params(self, model_name, prefix=''):
        """Get hyperparameter grid for each model"""
        params = {}
        
        if model_name == 'Random Forest':
            params = {
                f'{prefix}n_estimators': [100, 200],
                f'{prefix}max_depth': [10, 15, None],
                f'{prefix}min_samples_split': [2, 5],
                f'{prefix}min_samples_leaf': [1, 2]
            }
        elif model_name == 'Extra Trees':
            params = {
                f'{prefix}n_estimators': [100, 200],
                f'{prefix}max_depth': [10, 15, None],
                f'{prefix}min_samples_split': [2, 5]
            }
        elif model_name == 'Gradient Boosting':
            params = {
                f'{prefix}n_estimators': [100, 200],
                f'{prefix}learning_rate': [0.05, 0.1],
                f'{prefix}max_depth': [3, 5]
            }
        elif model_name == 'SVM':
            params = {
                f'{prefix}C': [1, 10, 100],
                f'{prefix}gamma': ['scale', 'auto']
            }
        elif model_name == 'Logistic Regression':
            params = {
                f'{prefix}C': [0.1, 1, 10],
                f'{prefix}penalty': ['l1', 'l2'],
                f'{prefix}solver': ['liblinear']
            }
        elif model_name == 'K-Nearest Neighbors':
            params = {
                f'{prefix}n_neighbors': [3, 5, 7],
                f'{prefix}weights': ['uniform', 'distance']
            }
        elif model_name == 'Decision Tree':
            params = {
                f'{prefix}max_depth': [10, 15, 20, None],
                f'{prefix}min_samples_split': [2, 5, 10],
                f'{prefix}min_samples_leaf': [1, 2, 3]
            }
        elif model_name == 'MLP Neural Network':
            params = {
                f'{prefix}hidden_layer_sizes': [(50,), (100,), (50, 50)],
                f'{prefix}alpha': [0.0001, 0.001, 0.01],
                f'{prefix}learning_rate': ['constant', 'adaptive']
            }
        elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
            params = {
                f'{prefix}n_estimators': [100, 200],
                f'{prefix}max_depth': [3, 5, 7],
                f'{prefix}learning_rate': [0.01, 0.1]
            }
        elif model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:
            params = {
                f'{prefix}n_estimators': [100, 200],
                f'{prefix}max_depth': [3, 5, 7],
                f'{prefix}learning_rate': [0.01, 0.1]
            }
        
        return params

    def train_and_evaluate_enhanced_models(self, test_size=0.2, random_state=42, max_models=15):
        """Train and evaluate enhanced models with comprehensive metrics"""
        print("\n" + "="*60)
        print("TRAINING AND EVALUATING ENHANCED MODELS")
        print("="*60)
        
        if not self.models:
            print("Please run setup_enhanced_models() first")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_encoded, self.y_encoded, test_size=test_size, 
            random_state=random_state, stratify=self.y_encoded
        )
        
        # Limit number of models to avoid excessive computation
        model_items = list(self.models.items())[:max_models]
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        
        for i, (model_name, config) in enumerate(model_items):
            print(f"\n[{i+1}/{len(model_items)}] Training {model_name}...")
            
            pipeline = config['pipeline']
            params = config['params']
            
            start_time = time.time()
            
            try:
                if params:
                    # Use reduced parameter grid for faster training
                    reduced_params = self._reduce_param_grid(params)
                    grid_search = GridSearchCV(
                        pipeline, reduced_params, cv=cv, scoring='accuracy', 
                        n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                else:
                    pipeline.fit(X_train, y_train)
                    best_model = pipeline
                    best_params = {}
                
                # Predictions
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                
                # Cross-validation score
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy')
                
                training_time = time.time() - start_time
                
                self.results[model_name] = {
                    'model': best_model,
                    'best_params': best_params,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'training_time': training_time,
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'classification_report': classification_report(y_test, y_pred)
                }
                
                print(f"  Accuracy: {accuracy:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | Time: {training_time:.1f}s")
                
            except Exception as e:
                print(f"  Error training {model_name}: {str(e)}")
                continue
        
        print(f"\nCompleted training {len(self.results)} models successfully")
        return self.results

    def _reduce_param_grid(self, params, max_combinations=12):
        """Reduce parameter grid size for faster training"""
        reduced_params = {}
        
        for param_name, param_values in params.items():
            if len(param_values) > 3:
                # Take first, middle, and last values
                reduced_values = [param_values[0], param_values[len(param_values)//2], param_values[-1]]
                reduced_params[param_name] = reduced_values
            else:
                reduced_params[param_name] = param_values
        
        return reduced_params

    def plot_enhanced_results(self):
        """Create comprehensive visualization of enhanced results"""
        if not self.results:
            print("No results to plot")
            return
        
        # Prepare data
        model_names = list(self.results.keys())
        metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Model Performance Comparison
        performance_data = []
        for metric in metrics:
            values = [self.results[name][metric] for name in model_names]
            performance_data.append(values)
        
        x_pos = np.arange(len(model_names))
        width = 0.15
        
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightgoldenrodyellow']
        
        for i, (metric, values) in enumerate(zip(metrics, performance_data)):
            axes[0, 0].bar(x_pos + i * width, values, width, label=metric.title(), 
                          color=colors[i], alpha=0.8)
        
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Enhanced Model Performance Comparison')
        axes[0, 0].set_xticks(x_pos + width * 2)
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Accuracy vs Training Time
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        training_times = [self.results[name]['training_time'] for name in model_names]
        
        scatter = axes[0, 1].scatter(training_times, accuracies, 
                                   c=range(len(model_names)), cmap='viridis', s=100, alpha=0.7)
        axes[0, 1].set_xlabel('Training Time (seconds)')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy vs Training Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add model name annotations
        for i, name in enumerate(model_names):
            axes[0, 1].annotate(name.split('_')[0], (training_times[i], accuracies[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 3. Cross-validation scores with error bars
        cv_means = [self.results[name]['cv_mean'] for name in model_names]
        cv_stds = [self.results[name]['cv_std'] for name in model_names]
        
        axes[1, 0].errorbar(range(len(model_names)), cv_means, yerr=cv_stds, 
                          fmt='o-', capsize=5, capthick=2, markersize=6)
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('CV Accuracy')
        axes[1, 0].set_title('Cross-Validation Scores with Standard Deviation')
        axes[1, 0].set_xticks(range(len(model_names)))
        axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Top performing models
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        top_10 = sorted_results[:10]
        
        top_names = [name for name, _ in top_10]
        top_accuracies = [result['accuracy'] for _, result in top_10]
        
        bars = axes[1, 1].barh(range(len(top_names)), top_accuracies, color='lightgreen', alpha=0.8)
        axes[1, 1].set_xlabel('Accuracy')
        axes[1, 1].set_ylabel('Models')
        axes[1, 1].set_title('Top 10 Performing Models')
        axes[1, 1].set_yticks(range(len(top_names)))
        axes[1, 1].set_yticklabels(top_names)
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        # Add accuracy values on bars
        for i, (bar, acc) in enumerate(zip(bars, top_accuracies)):
            axes[1, 1].text(acc + 0.005, i, f'{acc:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return fig

    def generate_comprehensive_report(self):
        """Generate a comprehensive performance report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print("="*80)
        
        if not self.results:
            print("No results available")
            return
        
        # Sort results by accuracy
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        print(f"\nTOTAL MODELS TRAINED: {len(self.results)}")
        print(f"BEST ACCURACY: {sorted_results[0][1]['accuracy']:.4f} ({sorted_results[0][0]})")
        
        # Performance summary table
        print(f"\n{'Rank':<4} {'Model':<30} {'Accuracy':<9} {'F1':<7} {'ROC-AUC':<8} {'CV Mean':<8} {'Time(s)':<8}")
        print("-" * 80)
        
        for i, (model_name, results) in enumerate(sorted_results[:15]):
            print(f"{i+1:<4} {model_name[:29]:<30} {results['accuracy']:<9.4f} "
                  f"{results['f1_score']:<7.4f} {results['roc_auc']:<8.4f} "
                  f"{results['cv_mean']:<8.4f} {results['training_time']:<8.1f}")
        
        # Best models by different metrics
        print(f"\nBEST MODELS BY DIFFERENT METRICS:")
        print("-" * 50)
        
        metrics_to_check = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']
        for metric in metrics_to_check:
            best_model = max(self.results.items(), key=lambda x: x[1][metric])
            print(f"Best {metric.upper()}: {best_model[1][metric]:.4f} ({best_model[0]})")
        
        # Model type analysis
        print(f"\nMODEL TYPE ANALYSIS:")
        print("-" * 30)
        
        pca_models = [name for name in self.results.keys() if 'PCA' in name]
        non_pca_models = [name for name in self.results.keys() if 'PCA' not in name]
        
        if pca_models:
            pca_avg_accuracy = np.mean([self.results[name]['accuracy'] for name in pca_models])
            print(f"PCA Models Average Accuracy: {pca_avg_accuracy:.4f} ({len(pca_models)} models)")
        
        if non_pca_models:
            non_pca_avg_accuracy = np.mean([self.results[name]['accuracy'] for name in non_pca_models])
            print(f"Non-PCA Models Average Accuracy: {non_pca_avg_accuracy:.4f} ({len(non_pca_models)} models)")
        
        # Improvement analysis
        print(f"\nIMPROVEMENT ANALYSIS:")
        print("-" * 25)
        
        baseline_accuracies = {
            'Random Forest': 0.725,
            'Extra Trees': 0.685,
            'Gradient Boosting': 0.755,
            'SVM': 0.7125,
            'Logistic Regression': 0.7225,
            'K-Nearest Neighbors': 0.65,
            'Decision Tree': 0.6375,
            'XGBoost': 0.7575,
            'LightGBM': 0.745
        }
        
        improvements = []
        for baseline_name, baseline_acc in baseline_accuracies.items():
            # Find best enhanced version
            enhanced_models = [name for name in self.results.keys() 
                             if baseline_name.replace(' ', '_').replace('-', '_') in name.replace('-', '_')]
            
            if enhanced_models:
                best_enhanced = max(enhanced_models, key=lambda x: self.results[x]['accuracy'])
                enhanced_acc = self.results[best_enhanced]['accuracy']
                improvement = enhanced_acc - baseline_acc
                improvements.append(improvement)
                
                print(f"{baseline_name}: {baseline_acc:.4f} → {enhanced_acc:.4f} "
                      f"(+{improvement:.4f}, {improvement/baseline_acc*100:+.1f}%)")
        
        if improvements:
            avg_improvement = np.mean(improvements)
            print(f"\nAVERAGE IMPROVEMENT: +{avg_improvement:.4f} ({avg_improvement/np.mean(list(baseline_accuracies.values()))*100:+.1f}%)")
        
        return sorted_results

    def save_enhanced_results(self, output_path='enhanced_blood_cancer_results.csv'):
        """Save comprehensive results to CSV"""
        print(f"\nSaving enhanced results to {output_path}...")
        
        if not self.results:
            print("No results to save")
            return
        
        # Prepare comprehensive results DataFrame
        results_list = []
        
        for model_name, result in self.results.items():
            results_list.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'F1_Score': result['f1_score'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'ROC_AUC': result['roc_auc'],
                'CV_Mean': result['cv_mean'],
                'CV_Std': result['cv_std'],
                'Training_Time': result['training_time'],
                'Best_Params': str(result['best_params'])
            })

        results_df = pd.DataFrame(results_list)
        results_df.to_csv(output_path, index=False) 
        print(f"Results saved to {output_path}")
        print("You can now analyze the results in 'enhanced_blood_cancer_results.csv'.")
        print("You can also visualize the results in the generated plots.")

        print("\n" + "="*60)
        print("ENHANCED BLOOD CANCER EDA COMPLETED")
        