import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


# --- 0. Reproducibility -------------------------------------------------------

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 1. Load Data -------------------------------------------------------------

file_name = "diabetes_012_health_indicators_BRFSS2015.csv"
df = pd.read_csv(file_name)

num_rows, num_cols = df.shape
print(f"Total rows: {num_rows}")
print(f"Total labels (columns): {num_cols}")
print(
    f"\nClass distribution:\n{df['Diabetes_012'].value_counts().sort_index()}")


# --- 2. Split Data (70% train / 15% val / 15% test, stratified) --------------

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


# --- 3. Feature Scaling -------------------------------------------------------

pt = PowerTransformer(method="yeo-johnson")
X_train_scaled = pt.fit_transform(X_train)
X_val_scaled = pt.transform(X_val)
X_test_scaled = pt.transform(X_test)


# --- 4. Reshape for LSTM (samples, timesteps, features) ----------------------
# LSTMs expect 3D input: (batch, sequence_length, input_size)
# Since this is tabular data (not time-series), we treat each feature
# as one timestep. Features are grouped by medical category so adjacent
# timesteps are related — this gives the LSTM meaningful sequential structure
# to learn from rather than arbitrary ordering.
#
# Group order:
#   [HighBP, HighChol, CholCheck]            -> cardiovascular indicators
#   [BMI]                                    -> body measurement
#   [Smoker, HvyAlcoholConsump]              -> lifestyle/substance use
#   [Stroke, HeartDiseaseorAttack]           -> serious conditions
#   [PhysActivity, Fruits, Veggies]          -> healthy behaviors
#   [AnyHealthcare, NoDocbcCost]             -> healthcare access
#   [GenHlth, MentHlth, PhysHlth, DiffWalk] -> general health status
#   [Sex, Age, Education, Income]            -> demographics

feature_order = [
    "HighBP", "HighChol", "CholCheck",
    "BMI",
    "Smoker", "HvyAlcoholConsump",
    "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies",
    "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
    "Sex", "Age", "Education", "Income"
]

X_train_ordered = pd.DataFrame(X_train_scaled, columns=X_train.columns)[
    feature_order].values
X_val_ordered = pd.DataFrame(X_val_scaled,   columns=X_val.columns)[
    feature_order].values
X_test_ordered = pd.DataFrame(X_test_scaled,  columns=X_test.columns)[
    feature_order].values

# Reshape: (samples, 21 timesteps, 1 feature per timestep)
X_train_seq = X_train_ordered.reshape(-1, 21, 1)
X_val_seq = X_val_ordered.reshape(-1, 21, 1)
X_test_seq = X_test_ordered.reshape(-1, 21, 1)


# --- 5. Convert to PyTorch Tensors -------------------------------------------

y_train_int = y_train.values.astype(int)
y_val_int = y_val.values.astype(int)
y_test_int = y_test.values.astype(int)

X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
X_val_t = torch.tensor(X_val_seq,   dtype=torch.float32)
X_test_t = torch.tensor(X_test_seq,  dtype=torch.float32)
y_train_t = torch.tensor(y_train_int, dtype=torch.long)
y_val_t = torch.tensor(y_val_int,   dtype=torch.long)
y_test_t = torch.tensor(y_test_int,  dtype=torch.long)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t,   y_val_t)
test_dataset = TensorDataset(X_test_t,  y_test_t)


# --- 6. Class Weights ---------------------------------------------------------
# FIX from v1: class 2 (diabetes) was getting 0 predictions because its
# base weight was only ~2.39 — far too low. The model just ignored it.
# Fix: boost class 2 by 5x and class 1 by 10x so all classes get predicted.
#
#   class 0 base weight: ~0.40  (majority, keep low)
#   class 1 base weight: ~18.3  x10 = ~183  (prediabetes, very rare)
#   class 2 base weight: ~2.39  x5  = ~12   (diabetes, was being ignored)

class_counts = np.bincount(y_train_int)
total = len(y_train_int)
weights = total / (len(class_counts) * class_counts.astype(float))
weights[1] *= 10.0   # class 1 (prediabetes) — extremely rare
weights[2] *= 5.0    # class 2 (diabetes)    — was ignored in v1

print(f"\nClass weights: {weights.round(4)}")
class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)


# --- 7. LSTM Model Definition ------------------------------------------------

class DiabetesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(DiabetesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]          # take last timestep only

        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out


# --- 8. Training & Evaluation Helpers ----------------------------------------

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    return np.array(all_labels), np.array(all_preds)


# --- 9. Hyperparameter Search ------------------------------------------------
# Wider search than v1: larger hidden sizes, more epochs, more patience
# Prediction distribution printed every 5 epochs so you can monitor
# whether class 2 is being predicted or still ignored

configs = [
    {"hidden_size": 128, "num_layers": 2,
        "dropout": 0.3, "lr": 1e-3, "batch_size": 512},
    {"hidden_size": 128, "num_layers": 2,
        "dropout": 0.4, "lr": 5e-4, "batch_size": 256},
    {"hidden_size": 256, "num_layers": 2,
        "dropout": 0.3, "lr": 1e-3, "batch_size": 512},
    {"hidden_size": 256, "num_layers": 2,
        "dropout": 0.4, "lr": 5e-4, "batch_size": 256},
    {"hidden_size": 256, "num_layers": 3,
        "dropout": 0.4, "lr": 1e-3, "batch_size": 512},
    {"hidden_size": 512, "num_layers": 2,
        "dropout": 0.4, "lr": 1e-3, "batch_size": 512},
    {"hidden_size": 512, "num_layers": 2,
        "dropout": 0.5, "lr": 5e-4, "batch_size": 256},
]

INPUT_SIZE = 1
NUM_CLASSES = 3
MAX_EPOCHS = 50
PATIENCE = 7

best_val_f1 = 0
best_config = None
best_model_state = None

print("\nStarting hyperparameter search...")
print("=" * 60)

for cfg in configs:
    print(f"\nConfig: {cfg}")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(
        val_dataset,   batch_size=cfg["batch_size"], shuffle=False)

    model = DiabetesLSTM(
        input_size=INPUT_SIZE,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_classes=NUM_CLASSES,
        dropout=cfg["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=False
    )

    best_epoch_f1 = 0
    patience_counter = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        loss = train_epoch(model, train_loader, criterion, optimizer)
        labels, preds = evaluate(model, val_loader)
        val_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        scheduler.step(val_f1)

        if val_f1 > best_epoch_f1:
            best_epoch_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            unique, counts = np.unique(preds, return_counts=True)
            pred_dist = dict(zip(unique.tolist(), counts.tolist()))
            print(
                f"  Epoch {epoch+1:2d} | Loss: {loss:.4f} | Val Macro F1: {val_f1:.4f} | Pred dist: {pred_dist}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"  Best val macro F1 for this config: {best_epoch_f1:.4f}")

    if best_epoch_f1 > best_val_f1:
        best_val_f1 = best_epoch_f1
        best_config = cfg
        best_model_state = best_state

print(f"\nBest config: {best_config}")
print(f"Best val macro F1: {best_val_f1:.4f}")


# --- 10. Final Evaluation -----------------------------------------------------

final_model = DiabetesLSTM(
    input_size=INPUT_SIZE,
    hidden_size=best_config["hidden_size"],
    num_layers=best_config["num_layers"],
    num_classes=NUM_CLASSES,
    dropout=best_config["dropout"]
).to(device)
final_model.load_state_dict(best_model_state)

train_loader_eval = DataLoader(
    train_dataset, batch_size=best_config["batch_size"], shuffle=False)
val_loader_eval = DataLoader(
    val_dataset,   batch_size=best_config["batch_size"], shuffle=False)
test_loader_eval = DataLoader(
    test_dataset,  batch_size=best_config["batch_size"], shuffle=False)

y_train_true, y_train_pred = evaluate(final_model, train_loader_eval)
y_val_true,   y_val_pred = evaluate(final_model, val_loader_eval)
y_test_true,  y_test_pred = evaluate(final_model, test_loader_eval)

train_acc = accuracy_score(y_train_true, y_train_pred)
val_acc = accuracy_score(y_val_true,   y_val_pred)
test_acc = accuracy_score(y_test_true,  y_test_pred)

train_f1 = f1_score(y_train_true, y_train_pred,
                    average="macro", zero_division=0)
val_f1 = f1_score(y_val_true,   y_val_pred,   average="macro", zero_division=0)
test_f1 = f1_score(y_test_true,  y_test_pred,
                   average="macro", zero_division=0)

print("\nFinal Results:")
print(f"Training accuracy:   {train_acc:.4f}")
print(f"Validation accuracy: {val_acc:.4f}")
print(f"Test accuracy:       {test_acc:.4f}")

print(f"\nTraining macro F1:   {train_f1:.4f}")
print(f"Validation macro F1: {val_f1:.4f}")
print(f"Test macro F1:       {test_f1:.4f}")

print("\nValidation Classification Report:")
print(classification_report(y_val_true, y_val_pred, zero_division=0))

print("Validation Confusion Matrix:")
print(confusion_matrix(y_val_true, y_val_pred))

print("\nTest Classification Report:")
print(classification_report(y_test_true, y_test_pred, zero_division=0))

print("Test Confusion Matrix:")
print(confusion_matrix(y_test_true, y_test_pred))


# --- 11. Confusion Matrix Plot (styled to match team slides) -----------------

def plot_confusion_matrix(y_true, y_pred, title, filename):
    from sklearn.metrics import classification_report
    import warnings
    warnings.filterwarnings("ignore")

    cm = confusion_matrix(y_true, y_pred)
    labels = ["No Diabetes", "Prediabetes", "Diabetes"]

    # Extract per-class precision, recall, f1 from report
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0)
    table_data = [
        ["0 (No diabetes)", f"{report['0']['precision']:.2f}",
         f"{report['0']['recall']:.2f}", f"{report['0']['f1-score']:.2f}"],
        ["1 (Prediabetes)", f"{report['1']['precision']:.2f}",
         f"{report['1']['recall']:.2f}", f"{report['1']['f1-score']:.2f}"],
        ["2 (Diabetes)",    f"{report['2']['precision']:.2f}",
         f"{report['2']['recall']:.2f}", f"{report['2']['f1-score']:.2f}"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [1.1, 1]})
    fig.patch.set_facecolor("#CCE9FF")

    # Left: heatmap
    ax1 = axes[0]
    ax1.set_facecolor("#CCE9FF")
    sns.heatmap(
        cm,
        annot=True,
        fmt=",",
        cmap="YlOrRd",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
        ax=ax1,
        annot_kws={"size": 13, "weight": "bold", "color": "white"},
        cbar_kws={"shrink": 0.8}
    )
    ax1.set_title(f"Confusion Matrix (LSTM)",
                  fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax1.set_ylabel("True Label",      fontsize=11, labelpad=8)
    ax1.tick_params(axis="x", labelsize=10, rotation=15)
    ax1.tick_params(axis="y", labelsize=10, rotation=0)

    # Right: precision/recall table
    ax2 = axes[1]
    ax2.set_facecolor("#CCE9FF")
    ax2.axis("off")
    ax2.text(0.5, 1.02, "Precision/Recall Class Data:",
             ha="center", va="bottom", fontsize=12,
             fontweight="bold", transform=ax2.transAxes)

    table = ax2.table(
        cellText=table_data,
        colLabels=["Class", "Precision", "Recall", "F1"],
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.2)

    for col in range(4):
        table[0, col].set_facecolor("#2B5C8A")
        table[0, col].set_text_props(color="white", fontweight="bold")
        table[0, col].set_edgecolor("white")

    row_colors = ["#FFFFFF", "#E8F4FD", "#FFFFFF"]
    for row in range(1, 4):
        for col in range(4):
            table[row, col].set_facecolor(row_colors[row - 1])
            table[row, col].set_edgecolor("#AAAAAA")
            if col == 0:
                table[row, col].set_text_props(fontweight="bold")

    plt.suptitle(title, fontsize=16, fontweight="bold",
                 color="#EA3D3D", y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight", facecolor="#CCE9FF")
    plt.show()
    print(f"Saved: {filename}")


plot_confusion_matrix(y_val_true,  y_val_pred,
                      "LSTM Validation Results", "lstm_val_confusion_matrix.png")
plot_confusion_matrix(y_test_true, y_test_pred,
                      "LSTM Test Results",       "lstm_test_confusion_matrix.png")
