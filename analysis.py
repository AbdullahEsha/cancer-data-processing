import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset - make sure the path is correct
df = pd.read_csv('public/blood_cancer_diseases_dataset.csv')

# Fill missing values
df['Genetic_Data(BCR-ABL, FLT3)'] = df['Genetic_Data(BCR-ABL, FLT3)'].fillna('Unknown')
df['Side_Effects'] = df['Side_Effects'].fillna('Unknown')


# Display basic information
# print("=== Dataset Overview ===")
# print(df.head())

# print("\n=== Basic Statistics ===")
# print(df.describe())

# print("\n=== Data Types ===")
print(df.columns.tolist())


# print("\n=== Missing Values ===")
# print(df.isnull().sum())


