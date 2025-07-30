## Activating the Virtual Environment

- Windows:

```bash
.\.venv\Scripts\activate
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

- Deactivating the Environment

```bash
deactivate
```

# Install dependencies

```bash
pip install -r requirements.txt
```

# Freeze dependencies

```bash
pip freeze > requirements.txt
```
## Class distribution: {'ALL': np.int64(333), 'AML': np.int64(333), 'CLL': np.int64(334)}
# Model Performance Metrics:
--------------------------------------------------------------------------------
Random Forest:
  Training Accuracy:  1.0000
  Test Accuracy:      0.9800
  Training Precision: 1.0000
  Test Precision:     0.9805
  Training Recall:    1.0000
  Test Recall:        0.9800
  Training F1:        1.0000
  Test F1:            0.9801
  Training AUC:       1.0000
  Test AUC:           0.9995

Gradient Boosting:
  Training Accuracy:  1.0000
  Test Accuracy:      0.9850
  Training Precision: 1.0000
  Test Precision:     0.9851
  Training Recall:    1.0000
  Test Recall:        0.9850
  Training F1:        1.0000
  Test F1:            0.9850
  Training AUC:       1.0000
  Test AUC:           0.9992

SVM:
  Training Accuracy:  1.0000
  Test Accuracy:      0.9850
  Training Precision: 1.0000
  Test Precision:     0.9851
  Training Recall:    1.0000
  Test Recall:        0.9850
  Training F1:        1.0000
  Test F1:            0.9850
  Training AUC:       1.0000
  Test AUC:           0.9997

Logistic Regression:
  Training Accuracy:  1.0000
  Test Accuracy:      0.9850
  Training Precision: 1.0000
  Test Precision:     0.9851
  Training Recall:    1.0000
  Test Recall:        0.9850
  Training F1:        1.0000
  Test F1:            0.9850
  Training AUC:       1.0000
  Test AUC:           0.9998

Best Model: Gradient Boosting
----------------------------------------
Classification Report:
              precision    recall  f1-score   support

         ALL     1.0000    0.9851    0.9925        67
         AML     0.9848    0.9848    0.9848        66
         CLL     0.9706    0.9851    0.9778        67

    accuracy                         0.9850       200
   macro avg     0.9851    0.9850    0.9850       200
weighted avg     0.9851    0.9850    0.9850       200

Confusion Matrix:
[[66  0  1]
 [ 0 65  1]
 [ 0  1 66]]
