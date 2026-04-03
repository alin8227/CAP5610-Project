import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

file_name = "diabetes_012_health_indicators_BRFSS2015.csv"
df = pd.read_csv(file_name)

X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"]

# Split dataset into train, test, val
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("\nDataset split:")
print(f"Train set: {len(X_train)} rows")
print(f"Validation set: {len(X_val)} rows")
print(f"Test set: {len(X_test)} rows")

pipeline = Pipeline([
    ('scalar', StandardScaler()),
    ('smote', SMOTE(random_state = 0)),
    ('svm', LinearSVC(random_state = 0))
])

class_weights = [None, 'balanced']

for w1 in range(10, 21):
    for w2 in [2, 3, 4, 5, 6]:
        class_weights.append({0: 1, 1: w1, 2: w2})
        
param_space = {
    'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'svm__max_iter': [100, 1000, 5000],
    'svm__class_weight': class_weights
}

GS = GridSearchCV(
    estimator = pipeline,
    param_grid = param_space,
    scoring = 'f1_macro',
    cv = 5, 
    n_jobs = -1,
    verbose = 2
)

GS.fit(X_train, y_train)

print("\nBest Parameters:", GS.best_params_)
print("Best Cross-Validation Macro F1:", GS.best_score_)
print("\n Techniques Model Used:", GS.best_estimator_)

# Train the svm model
best_params = GS.best_params_
svm_model = GS.best_estimator_

# Test the predictions using the validation set
y_pred_val = svm_model.predict(X_val)

print("\nValidation Confusion Matrix: ", confusion_matrix(y_val, y_pred_val))
print("Validation Classification Report: ", classification_report(y_val, y_pred_val))

print("\nValidation Accuracy:", accuracy_score(y_val, y_pred_val))
print("Validation F1:", f1_score(y_val, y_pred_val, average = 'macro'))

# Predict using the test set
y_pred_test = svm_model.predict(X_test)

print("\nTest Confusion Matrix: ", confusion_matrix(y_test, y_pred_test))
print("Test Classification Report: ", classification_report(y_test, y_pred_test))

print("\nTest Accuracy:", accuracy_score(y_test, y_pred_test))
print("Test F1:", f1_score(y_test, y_pred_test, average = 'macro'))