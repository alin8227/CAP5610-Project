import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# Load the dataset
file_name = "diabetes_012_health_indicators_BRFSS2015.csv"
df = pd.read_csv(file_name)

num_rows, num_cols = df.shape
print("Number of rows:", num_rows)
print("Number of columns:", num_cols)

# Split features and target
X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"]

# Split the dataset into train / validation / test split
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

# Combine train and validation for tuning
X_trainval = pd.concat([X_train, X_val], axis=0)
y_trainval = pd.concat([y_train, y_val], axis=0)

# Build pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", DecisionTreeClassifier(random_state=42))
])

# Class weights
class_weights = ["balanced"]

for w1 in range(5, 20):
    for w2 in [2, 3, 4, 5, 6]:
        class_weights.append({0: 1, 1: w1, 2: w2})

# Hyperparameter grid
param_grid = {
    "clf__criterion": ["gini", "entropy"],
    "clf__max_depth": [5, 10, 20, 30, None],
    "clf__min_samples_split": [2, 5, 10, 20],
    "clf__min_samples_leaf": [1, 2, 4, 8],
    "clf__class_weight": class_weights
}

# Grid search
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

# Best model
best_model = grid.best_estimator_

# Predictions
y_train_pred = best_model.predict(X_train)
y_val_pred = best_model.predict(X_val)
y_test_pred = best_model.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# Macro F1
train_macro_f1 = f1_score(y_train, y_train_pred, average="macro")
val_macro_f1 = f1_score(y_val, y_val_pred, average="macro")
test_macro_f1 = f1_score(y_test, y_test_pred, average="macro")

# Results
print("\nFinal Results:")
print(f"Training accuracy:   {train_acc:.4f}")
print(f"Validation accuracy: {val_acc:.4f}")
print(f"Test accuracy:       {test_acc:.4f}")

print(f"\nTraining macro F1:   {train_macro_f1:.4f}")
print(f"Validation macro F1: {val_macro_f1:.4f}")
print(f"Test macro F1:       {test_macro_f1:.4f}")

# Reports
print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))

print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))

print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))