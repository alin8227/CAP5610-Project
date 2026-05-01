import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Convert data into PyTorch tensors
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Transformer model for tabular data
class TabularTransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_layers, num_classes, dropout):
        super().__init__()

        # Embedding vector
        self.feature_embedding = nn.Linear(1, d_model)
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_features + 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Stack multiple encoder layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Add feature dimension
        x = x.unsqueeze(-1)

        # Convert each feature to embedding
        x = self.feature_embedding(x)

        # Expand CLS token to batch size
        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)

        # Prepare CLS token
        x = torch.cat([cls, x], dim=1)
        # Add positional encoding
        x = x + self.pos_embedding

        # Pass through transformer encoder
        x = self.transformer(x)

        # Use CLS token output for classification
        return self.classifier(x[:, 0])

# Training loop
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# Evaluation
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch)
            pred = torch.argmax(outputs, dim=1).cpu().numpy()

            preds.extend(pred)
            targets.extend(y_batch.numpy())

    return f1_score(targets, preds, average="macro"), np.array(preds), np.array(targets)

# Random hyperparameter sampler for random search
def sample_params():
    return {
        "d_model": random.choice([32, 64, 128]),
        "nhead": random.choice([2, 4, 8]),
        "num_layers": random.choice([1, 2, 3]),
        "dropout": random.choice([0.1, 0.2, 0.3]),
        "lr": random.choice([1e-3, 5e-4, 1e-4]),
        "batch_size": random.choice([128, 256])
    }

def main():

    # Load data
    df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

    X = df.drop("Diabetes_012", axis=1)
    y = df["Diabetes_012"]

    # Train / Val / Test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Feature scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Create PyTorch datasets
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    test_dataset = TabularDataset(X_test, y_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Handle class imbalance using weights
    classes = np.array([0, 1, 2])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # Hyperparameter tuning - random search
    best_f1 = -1
    best_params = None

    print("\nTuning...\n")

    for trial in range(20):
        params = sample_params()
        print("Trial", trial, params)

        model = TabularTransformer(
            num_features=X.shape[1],
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_layers=params["num_layers"],
            num_classes=3,
            dropout=params["dropout"]
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        criterion = nn.CrossEntropyLoss(weight=weights)

        train_loader = DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        val_loader = DataLoader(val_dataset, batch_size=256, num_workers=2)

        for _ in range(5):
            train(model, train_loader, optimizer, criterion, device)

        f1, _, _ = evaluate(model, val_loader, device)

        print("F1:", f1)

        if f1 > best_f1:
            best_f1 = f1
            best_params = params

    print("\nBEST PARAMS:", best_params)
    print("BEST F1:", best_f1)

    # Final model training with best parameters
    model = TabularTransformer(
        num_features=X.shape[1],
        d_model=best_params["d_model"],
        nhead=best_params["nhead"],
        num_layers=best_params["num_layers"],
        num_classes=3,
        dropout=best_params["dropout"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_loader = DataLoader(
        train_dataset,
        batch_size=best_params["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    print("\nFinal training...\n")

    for epoch in range(30):
        loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}: loss={loss:.4f}")

    # Evaluation
    train_loader_eval = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    _, y_train_pred, y_train_true = evaluate(model, train_loader_eval, device)
    _, y_val_pred, y_val_true = evaluate(model, val_loader, device)
    _, y_test_pred, y_test_true = evaluate(model, test_loader, device)

    train_acc = accuracy_score(y_train_true, y_train_pred)
    val_acc = accuracy_score(y_val_true, y_val_pred)
    test_acc = accuracy_score(y_test_true, y_test_pred)

    print("\nFINAL RESULTS")

    print(f"Train Accuracy: {train_acc:.4f} | Macro F1: {f1_score(y_train_true, y_train_pred, average='macro'):.4f}")
    print(f"Val Accuracy:   {val_acc:.4f} | Macro F1: {f1_score(y_val_true, y_val_pred, average='macro'):.4f}")
    print(f"Test Accuracy:  {test_acc:.4f} | Macro F1: {f1_score(y_test_true, y_test_pred, average='macro'):.4f}")

    print("\nValidation Classification Report:")
    print(classification_report(y_val_true, y_val_pred))

    print("\nTest Classification Report:")
    print(classification_report(y_test_true, y_test_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_true, y_test_pred))

if __name__ == "__main__":
    main()