import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data (as before)
df = pd.read_csv('public/blood_cancer_diseases_dataset.csv')
# ... [your preprocessing code here, including encoding and dropping columns] ...

X = df.drop('Cancer_Type(AML, ALL, CLL)', axis=1)
y = df['Cancer_Type(AML, ALL, CLL)']

# Feature selection
selector = SelectKBest(score_func=f_classif, k=min(10, X.shape[1]))
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_new)

# SMOTE for balancing classes
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# Gradient Boosting Model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
param_dist = {
    'max_depth': [3, 5, 7, 10],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2]
}
random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=10, scoring='accuracy', cv=5, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

best_xgb = random_search.best_estimator_
y_pred = best_xgb.predict(X_test)
train_pred = best_xgb.predict(X_train)

print(f"Training Accuracy: {accuracy_score(y_train, train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()