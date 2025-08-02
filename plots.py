import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Create a safe filename utility
def safe_filename(text):
    return re.sub(r'[^\w\-_.]', '_', text)

# Create a directory for saving plots
output_dir = 'public/plots'
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
df = pd.read_csv('public/sample_blood_cancer_diseases_dataset.csv')

# Check columns
print("Columns:", df.columns.tolist())

# Drop missing values for simplicity
df.dropna(inplace=True)

# Define target column and features
target_col = 'Cancer_Type(AML, ALL, CLL)'
features = ['Age', 'Total WBC count(/cumm)', 'Platelet Count( (/cumm)']

# Box plots to visualize distribution per cancer type
for feature in features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=target_col, y=feature, data=df)
    plt.title(f'{feature} vs {target_col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filename = f'{safe_filename(feature)}_vs_{safe_filename(target_col)}.png'
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()  # closes the figure instead of showing

# Pairplot (handle saving with tight layout)
pairplot = sns.pairplot(df[features + [target_col]], hue=target_col, diag_kind='kde')
pairplot.fig.suptitle("Pairplot: Feature Impact on Cancer Type", y=1.02)
pairplot_filename = os.path.join(output_dir, 'pairplot_feature_impact.png')
pairplot.savefig(pairplot_filename)
plt.close()

# Violin plots
for feature in features:
    plt.figure(figsize=(8, 5))
    sns.violinplot(x=target_col, y=feature, data=df)
    plt.title(f'{feature} Distribution by {target_col}')
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = f'{safe_filename(feature)}_distribution_by_{safe_filename(target_col)}.png'
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
