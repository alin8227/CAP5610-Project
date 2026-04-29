import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier

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

mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(solver = 'adam', random_state = 0, early_stopping = True, n_iter_no_change = 10))
])


param_space = {
    'mlp__hidden_layer_sizes': [(64,), (64,32), (128,)],
    'mlp__max_iter': [100, 200],
    'mlp__learning_rate_init': [0.0001, 0.001],
    'mlp__batch_size': [32, 64, 128],
    'mlp__alpha': [0.0001, 0.001]
}

GS = GridSearchCV(
    estimator = mlp_pipeline,
    param_grid = param_space,
    scoring = 'f1_macro',
    cv = 5, 
    n_jobs = 1,
    verbose = 2
)

GS.fit(X_train, y_train)

print("\nBest Parameters:", GS.best_params_)
print("Best Cross-Validation Macro F1:", GS.best_score_)
print("\n Techniques Model Used:", GS.best_estimator_)

# Train the mlp model
mlp_model = GS.best_estimator_

# Test the predictions using training
y_pred_train = mlp_model.predict(X_train)
print("\nTraining Confusion Matrix: ", confusion_matrix(y_train, y_pred_train))
print("Training Classification Report: ", classification_report(y_train, y_pred_train))

print("\nTraining Accuracy:", accuracy_score(y_train, y_pred_train))
print("Training F1:", f1_score(y_train, y_pred_train, average='macro'))

# Test the predictions using the validation set
y_pred_val = mlp_model.predict(X_val)

print("\nValidation Confusion Matrix: ", confusion_matrix(y_val, y_pred_val))
print("Validation Classification Report: ", classification_report(y_val, y_pred_val))

print("\nValidation Accuracy:", accuracy_score(y_val, y_pred_val))
print("Validation F1:", f1_score(y_val, y_pred_val, average = 'macro'))

# Predict using the test set
y_pred_test = mlp_model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred_test))
print("Test F1:", f1_score(y_test, y_pred_test, average = 'macro'))