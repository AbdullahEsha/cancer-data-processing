import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                           accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

class OptimizedCancerTypeClassifier:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.label_encoders = {}
        self.feature_selector = None
        self.best_model = None
        self.best_model_name = None
        self.target_encoder = None

    def load_data(self, file_path):
        # Load dataset from CSV
        df = pd.read_csv(file_path)

        # Check for required columns
        if 'Cancer_Type(AML, ALL, CLL)' not in df.columns:
            raise ValueError("Required column 'Cancer_Type(AML, ALL, CLL)' not found in the dataset.")

        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df

    def preprocess_data(self, df, target_column, test_size=0.2):
        # Validate target column
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found. Available columns: {df.columns.tolist()}")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Clean target variable
        print(f"Original target distribution: {y.value_counts().to_dict()}")

        # Remove invalid entries
        valid_cancer_types = ['AML', 'ALL', 'CLL', 'CML', 'Lymphoma', 'Multiple Myeloma']
        valid_mask = y.isin(valid_cancer_types)

        if not valid_mask.all():
            print(f"Removing {(~valid_mask).sum()} invalid target entries")
            X = X[valid_mask]
            y = y[valid_mask]

        # Enhanced preprocessing
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns

        # Encode categorical features
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        # Handle missing values with median/mode imputation
        for col in numerical_cols:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].median())

        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)

        # Remove constant features
        constant_features = X.columns[X.nunique() <= 1]
        if len(constant_features) > 0:
            print(f"Removing {len(constant_features)} constant features")
            X = X.drop(columns=constant_features)

        # Encode target
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # Robust scaling
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Advanced feature selection with RFE
        rf_selector = RandomForestClassifier(n_estimators=50, random_state=42)
        rfe = RFE(rf_selector, n_features_to_select=min(15, X_train.shape[1]))
        X_train_selected = rfe.fit_transform(X_train_scaled, y_train)
        X_test_selected = rfe.transform(X_test_scaled)
        self.feature_selector = rfe

        # SMOTE with Tomek links for better synthetic samples
        smote_tomek = ImbPipeline([
            ('smote', SMOTE(random_state=42, k_neighbors=3)),
            ('tomek', TomekLinks())
        ])

        X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_selected, y_train)

        print(f"Final training shape: {X_train_balanced.shape}")
        print(f"Final test shape: {X_test_selected.shape}")

        return X_train_balanced, X_test_selected, y_train_balanced, y_test

    def train_models(self, X_train, y_train):
        # Optimized model configurations with regularization
        model_configs = {
            'Random Forest': {
                'model': RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced',
                    oob_score=True
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [8, 12],
                    'min_samples_split': [10, 20],
                    'min_samples_leaf': [5, 10],
                    'max_features': ['sqrt', 0.7]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(
                    random_state=42,
                    n_iter_no_change=10,
                    validation_fraction=0.1
                ),
                'params': {
                    'n_estimators': [100, 150],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [4, 6],
                    'subsample': [0.8, 0.9],
                    'max_features': ['sqrt', 0.8]
                }
            },
            'SVM': {
                'model': SVC(
                    random_state=42,
                    probability=True,
                    class_weight='balanced'
                ),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf'],
                    'gamma': ['scale', 'auto']
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(
                    random_state=42,
                    max_iter=2000,
                    class_weight='balanced'
                ),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs']
                }
            }
        }

        # Use stratified 5-fold CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, config in model_configs.items():
            print(f"Training {name}...")

            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=cv,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X_train, y_train)
            self.models[name] = grid_search.best_estimator_
            print(f"{name} - CV Score: {grid_search.best_score_:.4f}")

        # Create ensemble
        ensemble = VotingClassifier([
            ('rf', self.models['Random Forest']),
            ('gb', self.models['Gradient Boosting']),
            ('svm', self.models['SVM']),
            ('lr', self.models['Logistic Regression'])
        ], voting='soft')

        ensemble.fit(X_train, y_train)
        self.models['Ensemble'] = ensemble

    def evaluate_models(self, X_train, X_test, y_train, y_test):
        results = {}

        for name, model in self.models.items():
            print(f"Evaluating {name}...")

            # Predictions
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)

            if hasattr(model, 'predict_proba'):
                y_pred_proba_test = model.predict_proba(X_test)
                y_pred_proba_train = model.predict_proba(X_train)
            else:
                y_pred_proba_test = None
                y_pred_proba_train = None

            # Metrics
            test_f1 = f1_score(y_test, y_pred_test, average='macro')
            train_f1 = f1_score(y_train, y_pred_train, average='macro')

            test_auc = None
            train_auc = None
            if y_pred_proba_test is not None:
                try:
                    test_auc = roc_auc_score(y_test, y_pred_proba_test, multi_class='ovr', average='macro')
                    train_auc = roc_auc_score(y_train, y_pred_proba_train, multi_class='ovr', average='macro')
                except:
                    pass

            # Overfitting score
            overfitting_score = train_f1 - test_f1

            results[name] = {
                'train_f1': train_f1,
                'test_f1': test_f1,
                'train_auc': train_auc,
                'test_auc': test_auc,
                'overfitting_score': overfitting_score,
                'y_pred_test': y_pred_test,
                'y_pred_proba_test': y_pred_proba_test
            }

        # Select best model based on lowest overfitting and highest test F1
        best_model = min(results.items(),
                        key=lambda x: (x[1]['overfitting_score'], -x[1]['test_f1']))

        self.best_model_name = best_model[0]
        self.best_model = self.models[self.best_model_name]

        return results

    def print_detailed_results(self, results, y_test):
        print("\n" + "="*80)
        print("OPTIMIZED CANCER TYPE CLASSIFIER RESULTS")
        print("="*80)

        for name, metrics in results.items():
            print(f"\n{name}:")
            print("-" * 50)
            print(f"  Test F1 Score:      {metrics['test_f1']:.4f}")
            print(f"  Train F1 Score:     {metrics['train_f1']:.4f}")
            print(f"  Overfitting Score:  {metrics['overfitting_score']:.4f}")
            if metrics['test_auc']:
                print(f"  Test AUC:           {metrics['test_auc']:.4f}")

        print(f"\n{'='*50}")
        print(f"BEST MODEL (Lowest Overfitting): {self.best_model_name}")
        print(f"{'='*50}")

        # Classification report for best model
        best_pred = results[self.best_model_name]['y_pred_test']
        print("\nClassification Report:")
        print(classification_report(y_test, best_pred,
                                  target_names=self.target_encoder.classes_))

def main():
    print("Optimized Cancer Type Classifier")
    print("Requires CSV file - No sample data generation")
    print("="*60)

    classifier = OptimizedCancerTypeClassifier()

    # Load data - MUST be from CSV
    try:
        df = classifier.load_data('public/blood_cancer_diseases_dataset.csv')
        print(f"Available columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error: {e}")
        return None, None

    # Determine target column
    target_col = 'Cancer_Type(AML, ALL, CLL)'  # Based on your output

    # Preprocess
    try:
        X_train, X_test, y_train, y_test = classifier.preprocess_data(df, target_col)
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None, None

    # Train models
    classifier.train_models(X_train, y_train)

    # Evaluate
    results = classifier.evaluate_models(X_train, X_test, y_train, y_test)

    # Print results
    classifier.print_detailed_results(results, y_test)

    return classifier, results

if __name__ == "__main__":
    classifier, results = main()