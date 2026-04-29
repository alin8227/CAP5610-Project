import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

# Load Data

file_name = "diabetes_012_health_indicators_BRFSS2015.csv"
df = pd.read_csv(file_name)

num_rows, num_cols = df.shape
print(f"Total rows: {num_rows}")
print(f"Total labels (columns): {num_cols}")
print(f"\nClass distribution:\n{df['Diabetes_012'].value_counts().sort_index()}")

# Split Data

X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\nTrain size:      {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size:       {len(X_test)}")

# Feature Scaling (PowerTransformer)


pt = PowerTransformer(method="yeo-johnson")
X_train_scaled = pt.fit_transform(X_train)
X_val_scaled   = pt.transform(X_val)
X_test_scaled  = pt.transform(X_test)

# Manual Class Weights via Priors GridSearch

priors = [None]  # None = use true class frequencies from training data
for p1 in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    for p2 in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        p0 = round(1.0 - p1 - p2, 4)
        if 0.05 <= p0 <= 0.70:
            priors.append((p0, p1, p2))  # tuples avoid Windows pickling error

param_grid = {
    "var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 0.5],
    "priors":        priors
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=GaussianNB(),
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=1,     # n_jobs=-1 causes PicklingError on Windows
    verbose=2
)

print("\nRunning GridSearchCV...")
grid.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid.best_params_}")
print(f"Best CV macro F1: {grid.best_score_:.4f}")

# Final Evaluation

best_model = grid.best_estimator_

y_pred_train = best_model.predict(X_train_scaled)
y_pred_val   = best_model.predict(X_val_scaled)
y_pred_test  = best_model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_pred_train)
val_acc   = accuracy_score(y_val,   y_pred_val)
test_acc  = accuracy_score(y_test,  y_pred_test)

train_f1  = f1_score(y_train, y_pred_train, average="macro")
val_f1    = f1_score(y_val,   y_pred_val,   average="macro")
test_f1   = f1_score(y_test,  y_pred_test,  average="macro")

print("\nFinal Results:")
print(f"Training accuracy:   {train_acc:.4f}")
print(f"Validation accuracy: {val_acc:.4f}")
print(f"Test accuracy:       {test_acc:.4f}")

print(f"\nTraining macro F1:   {train_f1:.4f}")
print(f"Validation macro F1: {val_f1:.4f}")
print(f"Test macro F1:       {test_f1:.4f}")

print("\nValidation Classification Report:")
print(classification_report(y_val, y_pred_val))

print("Validation Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_val))

print("\nTest Classification Report:")
print(classification_report(y_test, y_pred_test))

print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))
