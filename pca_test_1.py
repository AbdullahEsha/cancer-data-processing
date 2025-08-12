import pandas as pd
import os
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

# Additional libraries
from scipy import stats
from scipy.stats import boxcox, yeojohnson
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import combinations
import time

# read csv file from the public directory
df = pd.read_csv('public/sample_blood_cancer_diseases_dataset_numeric.csv')

# Display the first few rows of the dataset
print("Dataset loaded successfully. Here are the first few rows:")
print(df.head())

# Check for missing values
print("\nChecking for missing values in the dataset:")
print(df.isnull().sum())

# drop rows with missing values
df.dropna(inplace=True)

# again check for missing values after dropping
print("\nChecking for missing values after dropping rows:")
print(df.isnull().sum())

# Check the data types of the columns
print("\nData types of the columns:")
print(df.dtypes)

# Check the shape of the dataset
print("\nShape of the dataset:")
print(df.shape)

# convert the categorical columns to numerical values and create a new csv file with the above requirements
# Mapping for Cancer Type
cancer_type_mapping = {
    'AML': 0,
    'ALL': 1,
    'CLL': 2,
    'Lymphoma': 3,
    'Multiple Myeloma': 4,
    'CML': 5
}

# Mapping for Bone Marrow, Lymph Node, Lumbar Puncture
test_result_mapping = {
    'Positive': 1,
    'Negative': 0,
    'Not Done': -1
}

# Mapping for Genetic Data
genetic_data_mapping = {
    'BCR-ABL': 1,
    'FLT3': 0,
    'TP53': -1,
}

# Mapping for Diagnosis Result
diagnosis_result_mapping = {
    'Confirmed': 1,
    'Suspected': 0,
    'Ruled Out': -1
}

# Mapping for Gender
gender_mapping = {
    'Male': 1,
    'Female': 0,
    'Other': -1
}

# Mapping for Treatment Type
treatment_type_mapping = {
    'Targeted Therapy': 0,
    'Stem Cell Transplant': 1,
    'Chemotherapy': 2,
    'Immunotherapy': 3,
    'Radiation': 4
}

# Mapping for Serum Protein Electrophoresis
serum_protein_mapping = {
    'Normal': 0,
    'Abnormal': 1
}

# Mapping for Side Effects
side_effects_mapping = {
    'Mild': -1,
    'Moderate': 0,
    'Severe': 1
}

# Find max and min values for Total WBC count and Platelet Count
print("\nFinding max and min values for Total WBC count and Platelet Count:")

max_wbc = df['Total WBC count(/cumm)'].max()
min_wbc = df['Total WBC count(/cumm)'].min()
print(f"\nMax Total WBC count: {max_wbc}, Min Total WBC count: {min_wbc}")

max_platelet = df['Platelet Count(/cumm)'].max()
min_platelet = df['Platelet Count(/cumm)'].min()
print(f"Max Total Platelet count: {max_platelet}, Min Total Platelet count: {min_platelet}")

# Apply the mappings
df['Gender'] = df['Gender'].map(gender_mapping)
df['Cancer_Type(AML, ALL, CLL)'] = df['Cancer_Type(AML, ALL, CLL)'].map(cancer_type_mapping)
df['Treatment_Type(Chemotherapy, Radiation)'] = df['Treatment_Type(Chemotherapy, Radiation)'].map(treatment_type_mapping)
df['Bone Marrow Aspiration(Positive / Negative / Not Done)'] = df['Bone Marrow Aspiration(Positive / Negative / Not Done)'].map(test_result_mapping)
df['Lymph Node Biopsy(Positive / Negative / Not Done)'] = df['Lymph Node Biopsy(Positive / Negative / Not Done)'].map(test_result_mapping)
df['Serum Protein Electrophoresis(SPEP)(Normal / Abnormal)'] = df['Serum Protein Electrophoresis(SPEP)(Normal / Abnormal)'].map(serum_protein_mapping)
df['Lumbar Puncture(Spinal Tap)'] = df['Lumbar Puncture(Spinal Tap)'].map(test_result_mapping)
df['Genetic_Data(BCR-ABL, FLT3)'] = df['Genetic_Data(BCR-ABL, FLT3)'].map(genetic_data_mapping)
df['Side_Effects'] = df['Side_Effects'].map(side_effects_mapping)
df['Diagnosis_Result'] = df['Diagnosis_Result'].map(diagnosis_result_mapping)

# Save the new CSV file
output_path = 'public/processed_blood_cancer_dataset.csv'
df.to_csv(output_path, index=False)


# read the new csv file
df_processed = pd.read_csv(output_path)

# Display the first few rows of the processed dataset
print("\nProcessed dataset loaded successfully. Here are the first few rows:")
print(df_processed.head())
