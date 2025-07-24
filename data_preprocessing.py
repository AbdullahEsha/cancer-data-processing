import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

print("=== DATA PREPROCESSING FOR 4 KEY FEATURES ===")

# Load the original dataset
try:
    df = pd.read_csv('public/blood_cancer_diseases_dataset.csv')
    print(f"✅ Original dataset loaded successfully!")
    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
except FileNotFoundError:
    print("❌ Error: Could not find the original dataset file.")
    exit()

# Select only the 4 specified features
required_features = [
    "Age",
    "Platelet Count( (/cumm)",
    "Total WBC count(/cumm)",
    "Cancer_Type(AML, ALL, CLL)"
]

print(f"\n=== SELECTING REQUIRED FEATURES ===")
print(f"Required features: {required_features}")

# Check if all required features exist
missing_features = [col for col in required_features if col not in df.columns]
if missing_features:
    print(f"❌ Missing features: {missing_features}")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

# Select only the required features
df_selected = df[required_features].copy()
print(f"Selected dataset shape: {df_selected.shape}")

print(f"\n=== INITIAL DATA ANALYSIS ===")
print(f"Dataset info:")
print(df_selected.info())
print(f"\nMissing values per column:")
print(df_selected.isnull().sum())
print(f"\nTarget variable distribution:")
print(df_selected['Cancer_Type(AML, ALL, CLL)'].value_counts())

print(f"\n=== DATA CLEANING ===")

# 1. Drop rows with null values
print(f"Rows before dropping nulls: {len(df_selected)}")
df_clean = df_selected.dropna()
print(f"Rows after dropping nulls: {len(df_clean)}")

if len(df_clean) == 0:
    print("❌ No data remaining after dropping nulls!")
    exit()

# 2. Convert numeric columns to proper numeric types
numeric_features = ["Age", "Platelet Count( (/cumm)", "Total WBC count(/cumm)"]
print(f"\n=== CONVERTING TO NUMERIC ===")

for col in numeric_features:
    print(f"Processing column: {col}")
    print(f"  Original dtype: {df_clean[col].dtype}")
    
    # Convert to numeric, coercing any non-numeric values to NaN
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    print(f"  New dtype: {df_clean[col].dtype}")
    
    # Check for any NaN values created during conversion
    nan_count = df_clean[col].isnull().sum()
    if nan_count > 0:
        print(f"  Warning: {nan_count} non-numeric values converted to NaN")

# Drop any rows where numeric conversion failed
rows_before_numeric_clean = len(df_clean)
df_clean = df_clean.dropna()
rows_after_numeric_clean = len(df_clean)
print(f"Rows after numeric conversion cleanup: {rows_after_numeric_clean}")
if rows_before_numeric_clean != rows_after_numeric_clean:
    print(f"Dropped {rows_before_numeric_clean - rows_after_numeric_clean} rows due to non-numeric values")

# 3. Filter for main cancer types only (AML, ALL, CLL)
print(f"\n=== FILTERING CANCER TYPES ===")
main_cancer_types = ['AML', 'ALL', 'CLL']
print(f"Main cancer types to keep: {main_cancer_types}")

# Check current cancer type distribution
print(f"Cancer type distribution before filtering:")
print(df_clean['Cancer_Type(AML, ALL, CLL)'].value_counts())

# Filter for main cancer types
df_filtered = df_clean[df_clean['Cancer_Type(AML, ALL, CLL)'].isin(main_cancer_types)]
print(f"Rows after filtering cancer types: {len(df_filtered)}")

print(f"Cancer type distribution after filtering:")
print(df_filtered['Cancer_Type(AML, ALL, CLL)'].value_counts())

# 4. Basic data validation
print(f"\n=== DATA VALIDATION ===")
for col in numeric_features:
    print(f"{col}:")
    print(f"  Min: {df_filtered[col].min()}")
    print(f"  Max: {df_filtered[col].max()}")
    print(f"  Mean: {df_filtered[col].mean():.2f}")
    print(f"  Median: {df_filtered[col].median():.2f}")
    print(f"  Non-negative values: {(df_filtered[col] >= 0).sum()}/{len(df_filtered)}")

# 5. Remove any obviously invalid values (negative values for counts)
print(f"\n=== REMOVING INVALID VALUES ===")
count_features = ["Platelet Count( (/cumm)", "Total WBC count(/cumm)"]
age_feature = "Age"

# Check for negative values in count features
for col in count_features:
    negative_count = (df_filtered[col] < 0).sum()
    if negative_count > 0:
        print(f"Removing {negative_count} rows with negative {col}")
        df_filtered = df_filtered[df_filtered[col] >= 0]

# Check for unrealistic age values
unrealistic_age = ((df_filtered[age_feature] < 0) | (df_filtered[age_feature] > 120)).sum()
if unrealistic_age > 0:
    print(f"Removing {unrealistic_age} rows with unrealistic age values")
    df_filtered = df_filtered[(df_filtered[age_feature] >= 0) & (df_filtered[age_feature] <= 120)]

print(f"Final dataset shape after all cleaning: {df_filtered.shape}")

# 6. Final summary
print(f"\n=== FINAL PROCESSED DATA SUMMARY ===")
print(f"Shape: {df_filtered.shape}")
print(f"Features: {df_filtered.columns.tolist()}")
print(f"\nFinal statistics:")
print(df_filtered.describe())

print(f"\nFinal cancer type distribution:")
cancer_distribution = df_filtered['Cancer_Type(AML, ALL, CLL)'].value_counts()
print(cancer_distribution)
print(f"\nClass balance:")
for cancer_type, count in cancer_distribution.items():
    percentage = (count / len(df_filtered)) * 100
    print(f"  {cancer_type}: {count} samples ({percentage:.1f}%)")

# 7. Save the processed dataset
output_dir = 'public/processed_data'
os.makedirs(output_dir, exist_ok=True)
output_file = f'{output_dir}/clean_4_features_dataset.csv'

df_filtered.to_csv(output_file, index=False)
print(f"\n💾 Processed dataset saved to: {output_file}")

# 8. Save processing summary
processing_summary = {
    'original_shape': df.shape,
    'selected_features_shape': df_selected.shape,
    'after_dropping_nulls': len(df_clean),
    'after_numeric_conversion': rows_after_numeric_clean,
    'final_shape': df_filtered.shape,
    'features': df_filtered.columns.tolist(),
    'cancer_type_distribution': cancer_distribution.to_dict(),
    'rows_removed': {
        'null_values': df_selected.shape[0] - len(df_clean),
        'non_numeric': rows_before_numeric_clean - rows_after_numeric_clean,
        'non_main_cancer_types': len(df_clean) - len(df_filtered),
        'invalid_values': 0  # Will be updated if any invalid values were removed
    }
}

summary_file = f'{output_dir}/processing_summary.json'
import json
with open(summary_file, 'w') as f:
    json.dump(processing_summary, f, indent=2)

print(f"📊 Processing summary saved to: {summary_file}")
print(f"✅ Data preprocessing completed successfully!")
print(f"🎯 Ready for XGBoost analysis with {df_filtered.shape[0]} clean samples and {df_filtered.shape[1]} features")
