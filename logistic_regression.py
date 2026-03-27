import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

file_name = "diabetes_012_health_indicators_BRFSS2015.csv"
df = pd.read_csv(file_name)

num_rows, num_cols = df.shape
print(f"Total rows: {num_rows}")
print(f"Total labels (columns): {num_cols}")

X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"]

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

X_trainval = pd.concat([X_train, X_val], axis=0)
y_trainval = pd.concat([y_train, y_val], axis=0)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        random_state=42
    ))
])

class_weights = ["balanced"]

for w1 in range(5, 41):
    for w2 in [2, 3, 4, 5, 6]:
        class_weights.append({0: 1, 1: w1, 2: w2})

param_grid = {
    "clf__C": [0.0001, 0.01, 0.1, 0.05, 1, 3, 5],
    "clf__max_iter": [100, 500, 1000, 2000, 5000],
    "clf__class_weight": class_weights
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=3,
    n_jobs=-1,
    verbose=2
)

print("\nRunning hyperparameter tuning...")
grid.fit(X_trainval, y_trainval)

print("\nBest parameters:")
print(grid.best_params_)
print(f"Best CV macro F1: {grid.best_score_:.4f}")

best_model = grid.best_estimator_

y_train_pred = best_model.predict(X_train)
y_val_pred = best_model.predict(X_val)
y_test_pred = best_model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

train_macro_f1 = f1_score(y_train, y_train_pred, average="macro")
val_macro_f1 = f1_score(y_val, y_val_pred, average="macro")
test_macro_f1 = f1_score(y_test, y_test_pred, average="macro")

print("\nFinal Results:")
print(f"Training accuracy:   {train_acc:.4f}")
print(f"Validation accuracy: {val_acc:.4f}")
print(f"Test accuracy:       {test_acc:.4f}")

print(f"\nTraining macro F1:   {train_macro_f1:.4f}")
print(f"Validation macro F1: {val_macro_f1:.4f}")
print(f"Test macro F1:       {test_macro_f1:.4f}")

print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))

print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))

print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
