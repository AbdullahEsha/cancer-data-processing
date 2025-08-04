import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class CancerTypeClassifier:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_selector = None
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, file_path=None):
        try:
            # Try to read uploaded file first
            if file_path and 'fs' in dir(window):
                df = pd.read_csv(file_path)
                return df
        except:
            pass
        
        try:
            # Try to read from file system
            if file_path:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path)
                return df
        except:
            pass
        
        return self.create_sample_data()
    
    def create_sample_data(self):
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        data = np.random.randn(n_samples, n_features)
        
        aml_mask = np.arange(n_samples) < n_samples // 3
        data[aml_mask, :5] += np.random.normal(2, 0.5, (np.sum(aml_mask), 5))
        
        all_mask = (np.arange(n_samples) >= n_samples // 3) & (np.arange(n_samples) < 2 * n_samples // 3)
        data[all_mask, 5:10] += np.random.normal(-2, 0.5, (np.sum(all_mask), 5))
        
        cll_mask = np.arange(n_samples) >= 2 * n_samples // 3
        data[cll_mask, 10:15] += np.random.normal(1.5, 0.5, (np.sum(cll_mask), 5))
        
        labels = ['AML'] * (n_samples // 3) + ['ALL'] * (n_samples // 3) + ['CLL'] * (n_samples - 2 * (n_samples // 3))
        
        feature_names = [f'Gene_{i+1}' for i in range(n_features)]
        df = pd.DataFrame(data, columns=feature_names)
        df['Cancer_Type'] = labels
        
        return df
    
    def preprocess_data(self, df, target_column, test_size=0.2):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        # Encode categorical features
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Handle missing values separately for numerical and categorical
        for col in numerical_cols:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].median())
        
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)
        
        # Remove constant features
        constant_features = X.columns[X.nunique() <= 1]
        if len(constant_features) > 0:
            X = X.drop(columns=constant_features)
        
        # Encode target variable
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        self.target_encoder = target_encoder
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scaler = scaler
        
        # Feature selection
        selector = SelectKBest(f_classif, k=min(15, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        self.feature_selector = selector
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train)
        
        return X_train_balanced, X_test_selected, y_train_balanced, y_test
    
    def train_models(self, X_train, y_train):
        model_configs = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 5]
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [1, 10],
                    'kernel': ['rbf', 'linear']
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10]
                }
            }
        }
        
        for name, config in model_configs.items():
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=3, 
                scoring='f1_macro',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            self.models[name] = grid_search.best_estimator_
    
    def evaluate_models(self, X_train, X_test, y_train, y_test):
        results = {}
        
        for name, model in self.models.items():
            # Test predictions
            y_pred_test = model.predict(X_test)
            y_pred_proba_test = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Training predictions
            y_pred_train = model.predict(X_train)
            y_pred_proba_train = model.predict_proba(X_train) if hasattr(model, 'predict_proba') else None
            
            # Test metrics
            test_accuracy = accuracy_score(y_test, y_pred_test)
            test_precision = precision_score(y_test, y_pred_test, average='macro')
            test_recall = recall_score(y_test, y_pred_test, average='macro')
            test_f1 = f1_score(y_test, y_pred_test, average='macro')
            
            # Training metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            train_precision = precision_score(y_train, y_pred_train, average='macro')
            train_recall = recall_score(y_train, y_pred_train, average='macro')
            train_f1 = f1_score(y_train, y_pred_train, average='macro')
            
            # ROC AUC
            test_auc = None
            train_auc = None
            if y_pred_proba_test is not None:
                test_auc = roc_auc_score(y_test, y_pred_proba_test, multi_class='ovr', average='macro')
                train_auc = roc_auc_score(y_train, y_pred_proba_train, multi_class='ovr', average='macro')
            
            results[name] = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_precision': train_precision,
                'test_precision': test_precision,
                'train_recall': train_recall,
                'test_recall': test_recall,
                'train_f1': train_f1,
                'test_f1': test_f1,
                'train_auc': train_auc,
                'test_auc': test_auc,
                'y_pred_test': y_pred_test,
                'y_pred_proba_test': y_pred_proba_test
            }
        
        best_f1 = max(results.items(), key=lambda x: x[1]['test_f1'])
        self.best_model_name = best_f1[0]
        self.best_model = self.models[self.best_model_name]
        
        return results
    
    def print_metrics(self, results, y_train, y_test):
        # Create JSON output
        json_results = {}
        
        print("Model Performance Metrics:")
        print("-" * 80)
        
        for name, metrics in results.items():
            json_results[name] = {
                'training_accuracy': round(metrics['train_accuracy'], 4),
                'test_accuracy': round(metrics['test_accuracy'], 4),
                'training_precision': round(metrics['train_precision'], 4),
                'test_precision': round(metrics['test_precision'], 4),
                'training_recall': round(metrics['train_recall'], 4),
                'test_recall': round(metrics['test_recall'], 4),
                'training_f1': round(metrics['train_f1'], 4),
                'test_f1': round(metrics['test_f1'], 4)
            }
            
            if metrics['train_auc'] is not None:
                json_results[name]['training_auc'] = round(metrics['train_auc'], 4)
                json_results[name]['test_auc'] = round(metrics['test_auc'], 4)
            
            print(f"{name}:")
            print(f"  Training Accuracy:  {metrics['train_accuracy']:.4f}")
            print(f"  Test Accuracy:      {metrics['test_accuracy']:.4f}")
            print(f"  Training Precision: {metrics['train_precision']:.4f}")
            print(f"  Test Precision:     {metrics['test_precision']:.4f}")
            print(f"  Training Recall:    {metrics['train_recall']:.4f}")
            print(f"  Test Recall:        {metrics['test_recall']:.4f}")
            print(f"  Training F1:        {metrics['train_f1']:.4f}")
            print(f"  Test F1:            {metrics['test_f1']:.4f}")
            if metrics['train_auc'] is not None:
                print(f"  Training AUC:       {metrics['train_auc']:.4f}")
                print(f"  Test AUC:           {metrics['test_auc']:.4f}")
            print()
        
        # Add best model info
        json_results['best_model'] = {
            'name': self.best_model_name,
            'test_f1_score': round(results[self.best_model_name]['test_f1'], 4)
        }
        
        print(f"Best Model: {self.best_model_name}")
        print("-" * 40)
        
        best_y_pred = results[self.best_model_name]['y_pred_test']
        
        print("Classification Report:")
        report = classification_report(y_test, best_y_pred, 
                                     target_names=self.target_encoder.classes_,
                                     digits=4, output_dict=True)
        print(classification_report(y_test, best_y_pred, 
                                  target_names=self.target_encoder.classes_,
                                  digits=4))
        
        # Add classification report to JSON
        json_results['classification_report'] = report
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, best_y_pred)
        print(cm)
        
        # Add confusion matrix to JSON
        json_results['confusion_matrix'] = cm.tolist()
        
        # Print JSON output
        print("\nJSON Output:")
        print("-" * 40)
        print(json.dumps(json_results, indent=2))
        
        return json_results

def main():
    classifier = CancerTypeClassifier()
    
    # Try to load the uploaded file first
    try:
        df = pd.read_csv('public/blood_cancer_diseases_dataset.csv')
    except:
        df = classifier.load_data()
    
    if 'Cancer_Type' in df.columns:
        target_col = 'Cancer_Type'
    elif 'Cancer_Type(AML, ALL, CLL)' in df.columns:
        target_col = 'Cancer_Type(AML, ALL, CLL)'
    else:
        print("Available columns:", df.columns.tolist())
        target_col = input("Enter target column name: ")
    
    X_train, X_test, y_train, y_test = classifier.preprocess_data(df, target_col)
    
    classifier.train_models(X_train, y_train)
    
    results = classifier.evaluate_models(X_train, X_test, y_train, y_test)
    
    json_results = classifier.print_metrics(results, y_train, y_test)
    
    return classifier, results, json_results

if __name__ == "__main__":
    classifier, results, json_results = main()