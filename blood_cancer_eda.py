import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis libraries
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr, f_oneway
from matplotlib.gridspec import GridSpec

# Machine Learning libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                           roc_auc_score, precision_recall_fscore_support, roc_curve, auc)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# For XGBoost and LightGBM - install if not available
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

from datetime import datetime
from itertools import cycle

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

class BloodCancerClassifier:
    def __init__(self, csv_path=None):
        self.csv_path = csv_path
        self.df_raw = None
        self.df_clean = None
        self.X_encoded = None
        self.y_encoded = None
        self.label_encoder = None
        self.scaler = None
        self.models = {}
        self.results = {}
        
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
        print("DATA CLEANING AND PREPROCESSING")
    
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
        
        # Remove rows with missing genetic data if the column exists
        if 'genetic_data' in df.columns:
            initial_len = len(df)
            df = df.dropna(subset=['genetic_data']).copy()
            print(f"Removed {initial_len - len(df)} rows with missing genetic data")
        
        # Remove problematic header rows (data entry errors)
        problematic_values = [
            'Age', 'Cancer_Type(AML, ALL, CLL)', 'Total WBC count(/cumm)',
            'Platelet Count(/cumm)', 'Genetic_Data(BCR-ABL, FLT3)',
            'Bone Marrow Aspiration(Positive / Negative / Not Done)',
            'Lymph Node Biopsy(Positive / Negative / Not Done)',
            'Serum Protein Electrophoresis (SPEP)(Normal / Abnormal)',
            'Lumbar Puncture (Spinal Tap)', 'Diagnosis_Result'
        ]
        
        mask = True
        for value in problematic_values:
            mask = mask & ~df.isin([value]).any(axis=1)
        df = df[mask].copy()
        
        # Convert numeric columns
        numeric_columns = ['age', 'total_wbc_count', 'platelet_count']
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

    def exploratory_data_analysis(self):
        """Perform comprehensive EDA"""
        
        print("EXPLORATORY DATA ANALYSIS")
        
        
        if self.df_clean is None:
            print("Please run clean_data() first")
            return
        
        # Basic statistics
        print(f"Dataset Overview:")
        print(f"  Total Patients: {len(self.df_clean):,}")
        
        if 'age' in self.df_clean.columns:
            print(f"  Age Range: {self.df_clean['age'].min()} - {self.df_clean['age'].max()} years")
            print(f"  Mean Age: {self.df_clean['age'].mean():.1f} ± {self.df_clean['age'].std():.1f} years")
        
        if 'cancer_type' in self.df_clean.columns:
            print(f"\nCancer Type Distribution:")
            cancer_dist = self.df_clean['cancer_type'].value_counts()
            for cancer, count in cancer_dist.items():
                percentage = (count / len(self.df_clean)) * 100
                print(f"  {cancer}: {count:,} patients ({percentage:.1f}%)")
        
        # Create comprehensive visualization
        self.create_eda_visualizations()
        
        return self.df_clean.describe(include='all')

    def create_eda_visualizations(self):
        """Create comprehensive EDA visualizations"""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Age Distribution
        if 'age' in self.df_clean.columns:
            ax1 = fig.add_subplot(gs[0, 0])
            sns.histplot(data=self.df_clean, x='age', bins=20, kde=True, ax=ax1, alpha=0.7)
            ax1.set_title('Age Distribution', fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # 2. Cancer Type Distribution
        if 'cancer_type' in self.df_clean.columns:
            ax2 = fig.add_subplot(gs[0, 1])
            cancer_counts = self.df_clean['cancer_type'].value_counts()
            bars = ax2.bar(cancer_counts.index, cancer_counts.values, alpha=0.8)
            ax2.set_title('Cancer Type Distribution', fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{int(height)}', ha='center', va='bottom')
        
        # 3. WBC Count Distribution
        if 'total_wbc_count' in self.df_clean.columns:
            ax3 = fig.add_subplot(gs[0, 2])
            sns.histplot(data=self.df_clean, x='total_wbc_count', bins=30, kde=True, ax=ax3, alpha=0.7)
            ax3.set_title('WBC Count Distribution', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # 4. Platelet Count Distribution
        if 'platelet_count' in self.df_clean.columns:
            ax4 = fig.add_subplot(gs[0, 3])
            sns.histplot(data=self.df_clean, x='platelet_count', bins=30, kde=True, ax=ax4, alpha=0.7)
            ax4.set_title('Platelet Count Distribution', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        # 5. Age vs Cancer Type
        if 'age' in self.df_clean.columns and 'cancer_type' in self.df_clean.columns:
            ax5 = fig.add_subplot(gs[1, 0])
            sns.boxplot(data=self.df_clean, x='cancer_type', y='age', ax=ax5)
            ax5.set_title('Age Distribution by Cancer Type', fontweight='bold')
            ax5.tick_params(axis='x', rotation=45)
        
        # 6. Genetic Data Distribution
        if 'genetic_data' in self.df_clean.columns:
            ax6 = fig.add_subplot(gs[1, 1])
            genetic_counts = self.df_clean['genetic_data'].value_counts()
            ax6.pie(genetic_counts.values, labels=genetic_counts.index, autopct='%1.1f%%')
            ax6.set_title('Genetic Data Distribution', fontweight='bold')
        
        # 7. Laboratory Values by Cancer Type
        if all(col in self.df_clean.columns for col in ['cancer_type', 'total_wbc_count']):
            ax7 = fig.add_subplot(gs[1, 2])
            sns.violinplot(data=self.df_clean, x='cancer_type', y='total_wbc_count', ax=ax7)
            ax7.set_title('WBC Count by Cancer Type', fontweight='bold')
            ax7.tick_params(axis='x', rotation=45)
        
        # 8. Correlation heatmap for numeric variables
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            ax8 = fig.add_subplot(gs[1, 3])
            correlation_matrix = self.df_clean[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax8)
            ax8.set_title('Correlation Matrix', fontweight='bold')
        
        plt.suptitle('Comprehensive Blood Cancer Dataset Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def feature_engineering(self):
        """Create advanced features for better model performance"""
        print("FEATURE ENGINEERING")
        
        if self.df_clean is None:
            print("Please run clean_data() first")
            return
        
        df_enhanced = self.df_clean.copy()
        
        # Risk factors
        risk_factors = pd.DataFrame()
        if 'age' in df_enhanced.columns:
            risk_factors['age_risk'] = (df_enhanced['age'] > 65).astype(int)
        
        if 'total_wbc_count' in df_enhanced.columns:
            risk_factors['wbc_risk'] = (df_enhanced['total_wbc_count'] > 75000).astype(int)
            df_enhanced['log_wbc'] = np.log1p(df_enhanced['total_wbc_count'])
            if 'age' in df_enhanced.columns:
                df_enhanced['wbc_to_age_ratio'] = df_enhanced['total_wbc_count'] / (df_enhanced['age'] + 1)
        
        if 'platelet_count' in df_enhanced.columns:
            risk_factors['platelet_low_risk'] = (df_enhanced['platelet_count'] < 100000).astype(int)
            risk_factors['platelet_high_risk'] = (df_enhanced['platelet_count'] > 350000).astype(int)
            df_enhanced['log_platelet'] = np.log1p(df_enhanced['platelet_count'])
            if 'age' in df_enhanced.columns:
                df_enhanced['platelet_to_age_ratio'] = df_enhanced['platelet_count'] / (df_enhanced['age'] + 1)
        
        if 'genetic_data' in df_enhanced.columns:
            risk_factors['genetic_risk'] = df_enhanced['genetic_data'].isin(['FLT3', 'TP53']).astype(int)
        
        if 'bone_marrow_aspiration' in df_enhanced.columns:
            risk_factors['bma_positive'] = (df_enhanced['bone_marrow_aspiration'] == 'Positive').astype(int)
        
        if 'lymph_node_biopsy' in df_enhanced.columns:
            risk_factors['lnb_positive'] = (df_enhanced['lymph_node_biopsy'] == 'Positive').astype(int)
        
        if 'spep_result' in df_enhanced.columns:
            risk_factors['spep_abnormal'] = (df_enhanced['spep_result'] == 'Abnormal').astype(int)
        
        # Calculate total risk score
        risk_factors['total_risk_score'] = risk_factors.sum(axis=1)
        
        # Add risk factors to enhanced dataframe
        for col in risk_factors.columns:
            df_enhanced[col] = risk_factors[col]
        
        # Age groups
        if 'age' in df_enhanced.columns:
            df_enhanced['age_group'] = pd.cut(df_enhanced['age'], 
                                            bins=[0, 30, 50, 65, 100], 
                                            labels=['Young', 'Middle', 'Senior', 'Elderly'])
        
        # Interaction features
        if 'genetic_data' in df_enhanced.columns and 'bone_marrow_aspiration' in df_enhanced.columns:
            df_enhanced['genetic_bma_interaction'] = (
                df_enhanced['genetic_data'].isin(['FLT3', 'TP53']) &
                (df_enhanced['bone_marrow_aspiration'] == 'Positive')
            ).astype(int)
        
        # Lab complexity score
        lab_features = ['total_wbc_count', 'platelet_count']
        available_lab_features = [col for col in lab_features if col in df_enhanced.columns]
        
        if available_lab_features:
            complexity_score = 0
            for col in available_lab_features:
                if col == 'total_wbc_count':
                    complexity_score += (df_enhanced[col] > df_enhanced[col].quantile(0.75)).astype(int)
                elif col == 'platelet_count':
                    complexity_score += (df_enhanced[col] < df_enhanced[col].quantile(0.25)).astype(int)
                    complexity_score += (df_enhanced[col] > df_enhanced[col].quantile(0.75)).astype(int)
            
            df_enhanced['lab_complexity_score'] = complexity_score
        
        self.df_enhanced = df_enhanced
        print(f"Enhanced dataset shape: {self.df_enhanced.shape}")
        print(f"New features added: {len(self.df_enhanced.columns) - len(self.df_clean.columns)}")

        # Advanced Feature Engineering for Maximum Accuracy
        
        # 1. Polynomial Features for numeric variables
        
        numeric_cols = ['age', 'total_wbc_count', 'platelet_count', 'log_wbc', 'log_platelet']
        available_numeric = [col for col in numeric_cols if col in df_enhanced.columns]
        
        if len(available_numeric) >= 2:
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            poly_features = poly.fit_transform(df_enhanced[available_numeric])
            poly_feature_names = [f"poly_{name}" for name in poly.get_feature_names_out(available_numeric)]
            
            # Add only the most important polynomial features to avoid overfitting
            for i, name in enumerate(poly_feature_names[len(available_numeric):]):  # Skip original features
                df_enhanced[name] = poly_features[:, len(available_numeric) + i]
        
        # 2. Advanced ratio features
        if 'total_wbc_count' in df_enhanced.columns and 'platelet_count' in df_enhanced.columns:
            df_enhanced['wbc_platelet_ratio'] = df_enhanced['total_wbc_count'] / (df_enhanced['platelet_count'] + 1)
            df_enhanced['platelet_wbc_ratio'] = df_enhanced['platelet_count'] / (df_enhanced['total_wbc_count'] + 1)
        
        # 3. Binning continuous variables for better pattern capture
        if 'age' in df_enhanced.columns:
            df_enhanced['age_decade'] = (df_enhanced['age'] // 10) * 10
            df_enhanced['age_binned'] = pd.cut(df_enhanced['age'], 
                                             bins=[0, 25, 40, 55, 70, 100], 
                                             labels=['very_young', 'young', 'middle', 'senior', 'elderly'])
        
        if 'total_wbc_count' in df_enhanced.columns:
            df_enhanced['wbc_category'] = pd.cut(df_enhanced['total_wbc_count'],
                                               bins=[0, 4000, 11000, 50000, float('inf')],
                                               labels=['low', 'normal', 'high', 'very_high'])
        
        if 'platelet_count' in df_enhanced.columns:
            df_enhanced['platelet_category'] = pd.cut(df_enhanced['platelet_count'],
                                                    bins=[0, 100000, 350000, float('inf')],
                                                    labels=['low', 'normal', 'high'])
        
        # 4. Medical severity scores
        severity_score = 0
        if 'age' in df_enhanced.columns:
            severity_score += (df_enhanced['age'] > 60).astype(int) * 2
        if 'total_wbc_count' in df_enhanced.columns:
            severity_score += (df_enhanced['total_wbc_count'] > 100000).astype(int) * 3
            severity_score += (df_enhanced['total_wbc_count'] < 4000).astype(int) * 2
        if 'platelet_count' in df_enhanced.columns:
            severity_score += (df_enhanced['platelet_count'] < 50000).astype(int) * 3
        
        df_enhanced['medical_severity_score'] = severity_score
        
        # 5. Feature interactions with genetic data
        if 'genetic_data' in df_enhanced.columns:
            for test_col in ['bone_marrow_aspiration', 'lymph_node_biopsy', 'spep_result']:
                if test_col in df_enhanced.columns:
                    df_enhanced[f'genetic_{test_col}_combo'] = (
                        df_enhanced['genetic_data'].astype(str) + '_' + 
                        df_enhanced[test_col].astype(str)
                    )
        
        # 6. Statistical transformations
        if 'total_wbc_count' in df_enhanced.columns:
            df_enhanced['wbc_zscore'] = (df_enhanced['total_wbc_count'] - df_enhanced['total_wbc_count'].mean()) / df_enhanced['total_wbc_count'].std()
            df_enhanced['wbc_sqrt'] = np.sqrt(df_enhanced['total_wbc_count'])
        
        if 'platelet_count' in df_enhanced.columns:
            df_enhanced['platelet_zscore'] = (df_enhanced['platelet_count'] - df_enhanced['platelet_count'].mean()) / df_enhanced['platelet_count'].std()
            df_enhanced['platelet_sqrt'] = np.sqrt(df_enhanced['platelet_count'])
        
        print(f"Advanced features added. Total features now: {len(df_enhanced.columns)}")
        self.df_enhanced = df_enhanced
        
        return self.df_enhanced.head()

    def prepare_ml_features(self):
        """Prepare features for machine learning"""
        print("PREPARING FEATURES FOR MACHINE LEARNING")
        
        if not hasattr(self, 'df_enhanced'):
            print("Please run feature_engineering() first")
            return
        
        # Define target column
        target_col = 'cancer_type'
        if target_col not in self.df_enhanced.columns:
            print(f"Target column '{target_col}' not found!")
            return
        
        # Select feature columns (exclude target)
        feature_columns = [col for col in self.df_enhanced.columns if col != target_col]
        
        # Remove any remaining non-feature columns if needed
        exclude_cols = ['diagnosis_result']  # Add other columns to exclude if needed
        feature_columns = [col for col in feature_columns if col not in exclude_cols]
        
        X = self.df_enhanced[feature_columns].copy()
        y = self.df_enhanced[target_col].copy()
        
        # Handle categorical variables
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
    
    def feature_selection(self, top_k_features=50):
        """Select the most important features to improve model performance"""
        print("PERFORMING FEATURE SELECTION")
        
        from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
        from sklearn.ensemble import RandomForestClassifier
        
        if self.X_encoded is None or self.y_encoded is None:
            print("Please run prepare_ml_features() first")
            return
        
        # Method 1: Random Forest Feature Importance
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_selector.fit(self.X_encoded, self.y_encoded)
        feature_importance = pd.DataFrame({
            'feature': self.X_encoded.columns,
            'importance': rf_selector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Method 2: Mutual Information
        mi_scores = mutual_info_classif(self.X_encoded, self.y_encoded, random_state=42)
        mi_importance = pd.DataFrame({
            'feature': self.X_encoded.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        # Method 3: Statistical tests (for numerical features)
        try:
            f_scores = f_classif(self.X_encoded, self.y_encoded)[0]
            f_importance = pd.DataFrame({
                'feature': self.X_encoded.columns,
                'f_score': f_scores
            }).sort_values('f_score', ascending=False)
        except:
            f_importance = feature_importance.copy()
        
        # Combine rankings (ensemble feature selection)
        feature_scores = pd.DataFrame({'feature': self.X_encoded.columns})
        feature_scores['rf_rank'] = feature_scores['feature'].map(
            {f: i for i, f in enumerate(feature_importance['feature'])}
        )
        feature_scores['mi_rank'] = feature_scores['feature'].map(
            {f: i for i, f in enumerate(mi_importance['feature'])}
        )
        feature_scores['f_rank'] = feature_scores['feature'].map(
            {f: i for i, f in enumerate(f_importance['feature'])}
        )
        
        # Calculate combined score (lower is better)
        feature_scores['combined_rank'] = (
            feature_scores['rf_rank'] + 
            feature_scores['mi_rank'] + 
            feature_scores['f_rank']
        ) / 3
        
        feature_scores = feature_scores.sort_values('combined_rank')
        
        # Select top k features
        selected_features = feature_scores.head(min(top_k_features, len(feature_scores)))['feature'].tolist()
        
        print(f"Selected {len(selected_features)} most important features out of {len(self.X_encoded.columns)}")
        print("Top 10 selected features:")
        for i, feature in enumerate(selected_features[:10]):
            print(f"  {i+1}. {feature}")
        
        # Update encoded features
        self.X_encoded = self.X_encoded[selected_features]
        self.feature_importance = feature_importance
        
        return selected_features


    def setup_models(self):
        """Define enhanced models with optimized hyperparameters for maximum accuracy"""
        models_config = {
            'XGBoost_Optimized': {
                'model': XGBClassifier(random_state=42, eval_metric='mlogloss'),
                'params': {
                    'n_estimators': [300, 500, 800],
                    'max_depth': [4, 6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'min_child_weight': [1, 3, 5],
                    'gamma': [0, 0.1, 0.2],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [1, 1.5, 2]
                } if XGBOOST_AVAILABLE else {},
                'use_scaled': False
            },
            'LightGBM_Optimized': {
                'model': LGBMClassifier(random_state=42, verbose=-1),
                'params': {
                    'n_estimators': [300, 500, 800],
                    'max_depth': [4, 6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'min_child_samples': [10, 20, 30],
                    'num_leaves': [31, 50, 100],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [0, 0.1, 0.5]
                } if LIGHTGBM_AVAILABLE else {},
                'use_scaled': False
            },
            'Random Forest Ultra': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [300, 500, 800],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 3, 5],
                    'min_samples_leaf': [1, 2, 3],
                    'max_features': ['auto', 'sqrt', 'log2', None],
                    'bootstrap': [True, False],
                    'class_weight': ['balanced', None]
                },
                'use_scaled': False
            },
            'Extra Trees Ultra': {
                'model': ExtraTreesClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [300, 500, 800],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 3, 5],
                    'min_samples_leaf': [1, 2, 3],
                    'max_features': ['auto', 'sqrt', 'log2', None],
                    'bootstrap': [True, False],
                    'class_weight': ['balanced', None]
                },
                'use_scaled': False
            },
            'Gradient Boosting Ultra': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15],
                    'max_depth': [3, 4, 5, 6],
                    'subsample': [0.8, 0.9, 1.0],
                    'min_samples_split': [2, 3, 5],
                    'min_samples_leaf': [1, 2, 3],
                    'max_features': ['auto', 'sqrt', 'log2']
                },
                'use_scaled': False
            },
            'SVM_Optimized': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100, 1000],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'kernel': ['rbf', 'poly'],
                    'degree': [2, 3, 4],  # Only used for poly kernel
                    'class_weight': ['balanced', None]
                },
                'use_scaled': True
            },
            'Neural Network': {
                'model': MLPClassifier(random_state=42, max_iter=2000),
                'params': {
                    'hidden_layer_sizes': [(100,), (200,), (100, 50), (200, 100), (300, 150, 75)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive'],
                    'learning_rate_init': [0.001, 0.01, 0.1]
                },
                'use_scaled': True
            }
        }
        
        # Remove models with empty params (when libraries not available)
        self.models = {k: v for k, v in models_config.items() if v['params']}
        print(f"Enhanced models set up with {len(self.models)} configurations")
        return self.models
    
    def train_and_evaluate_models(self, test_size=0.2, random_state=42):
        """Enhanced training with better cross-validation and evaluation"""
        print("TRAINING AND EVALUATING ENHANCED MODELS")
        
        if self.X_encoded is None or self.y_encoded is None:
            print("Please run prepare_ml_features() first")
            return
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_encoded, self.y_encoded, test_size=test_size, 
            random_state=random_state, stratify=self.y_encoded
        )
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Enhanced cross-validation
        cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=random_state)
        
        for model_name, config in self.models.items():
            print(f"\nTraining {model_name}...")
            
            model = config['model']
            params = config['params']
            use_scaled = config['use_scaled']
            
            # Prepare data
            if use_scaled:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                X_train_model, X_test_model = X_train_scaled, X_test_scaled
            else:
                X_train_model, X_test_model = X_train, X_test
            
            # Enhanced Grid Search with more robust scoring
            grid_search = GridSearchCV(
                model, params, cv=cv, 
                scoring='accuracy',  # You can also try 'f1_macro' for imbalanced classes
                n_jobs=-1, 
                verbose=0,
                return_train_score=True
            )
            
            grid_search.fit(X_train_model, y_train)
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(X_test_model)
            y_pred_proba = best_model.predict_proba(X_test_model)
            
            # Enhanced evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Cross-validation scores
            cv_scores = cross_val_score(best_model, X_train_model, y_train, cv=cv, scoring='accuracy')
            
            # Calculate class-wise metrics
            precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
            
            # ROC AUC (multiclass)
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
            except:
                roc_auc = 0.0
            
            # Store comprehensive results
            self.results[model_name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'confusion_matrix': conf_matrix,
                'classification_report': classification_report(y_test, y_pred, 
                                                             target_names=self.label_encoder.classes_),
                'roc_auc': roc_auc,
                'precision_per_class': precision,
                'recall_per_class': recall,
                'f1_per_class': f1,
                'support_per_class': support,
                'best_cv_score': grid_search.best_score_
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"  ROC AUC: {roc_auc:.4f}")
        
        # Print best performing model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        print(f"\nBest performing model: {best_model_name}")
        print(f"Best accuracy: {self.results[best_model_name]['accuracy']:.4f}")
        
        return self.results
    
    def plot_model_performance(self):
        """Plot model performance metrics"""
        print("PLOTTING MODEL PERFORMANCE")
    
        if not self.results:
            print("No results to plot. Please run train_and_evaluate_models() first.")
            return
        
        # Prepare data for plotting
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        roc_aucs = [self.results[name]['roc_auc'] for name in model_names]
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot accuracy
        ax1.bar(model_names, accuracies, color='skyblue', alpha=0.7, label='Accuracy')
        ax1.set_ylabel('Accuracy', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')
        
        # Create a second y-axis for ROC AUC
        ax2 = ax1.twinx()
        ax2.plot(model_names, roc_aucs, color='orange', marker='o', label='ROC AUC')
        ax2.set_ylabel('ROC AUC', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # Add titles and labels
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Models')
        
        # Add legend
        fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85), bbox_transform=ax1.transAxes)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        print("Model performance plotted successfully.")
        return fig
    
    def save_results(self, output_path='public/test_1/blood_cancer_results.csv'):
        """Save results to a CSV file"""
        print("SAVING RESULTS")

        if not self.results:
            print("No results to save. Please run train_and_evaluate_models() first.")
            return
        
        # Prepare DataFrame for results
        results_list = []
        for model_name, result in self.results.items():
            results_list.append({
                'Model': model_name,
                'Best Parameters': str(result['best_params']),
                'Accuracy': result['accuracy'],
                'ROC AUC': result['roc_auc'],
                'Confusion Matrix': str(result['confusion_matrix']),
                'Classification Report': result['classification_report']
            })
        
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(output_path, index=False)
        
        print(f"Results saved to {output_path}")
        return output_path
    
    def run_all(self, csv_path=None, output_path='blood_cancer_results.csv'):
        """Run all steps in sequence"""
        print("RUNNING ALL STEPS")
        self.load_data()
        self.clean_data()
        self.exploratory_data_analysis()
        self.feature_engineering()
        self.prepare_ml_features()
        self.setup_models()
        self.train_and_evaluate_models()
        self.plot_model_performance()
        self.save_results(output_path)
        
        print("All steps completed successfully.")
        return output_path
    
if __name__ == "__main__":
    classifier = BloodCancerClassifier(csv_path='public/blood_cancer_diseases_dataset.csv')
    results_file = classifier.run_all(output_path='public/test_1/blood_cancer_results.csv')
    print(f"Results saved to {results_file}")