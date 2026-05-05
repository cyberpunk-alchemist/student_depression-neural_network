import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


# ==================================================
# 1. Reproducibility
# ==================================================
SEED = 676767
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# If CUDA is used later:
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================================================
# 2. Load data
# ==================================================
df = pd.read_csv("Data/Data_processed.csv")

# Explicit targets
target_cols = ["Have you ever had suicidal thoughts ?", "Depression"]

# Features and targets
X_df = df.drop(columns=target_cols)
y_df = df[target_cols]

X = X_df.to_numpy(dtype=np.float32)
y = y_df.to_numpy(dtype=np.float32)


# ==================================================
# 3. Train / validation / test split
# ==================================================
# First split off test set
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# Then split training into train + validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=SEED
)


# ==================================================
# 4. Scaling
# ==================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)


# ==================================================
# 5. Tensors and loaders
# ==================================================
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# ==================================================
# 6. Class imbalance handling: pos_weight
# ==================================================
# For BCEWithLogitsLoss, pos_weight_j = N_negative_j / N_positive_j
pos_counts = y_train.sum(axis=0)
neg_counts = len(y_train) - pos_counts

# Avoid division by zero
pos_weight = torch.tensor(
    neg_counts / np.clip(pos_counts, 1, None),
    dtype=torch.float32,
    device=device
)


# ==================================================
# 7. Define model
# ==================================================
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(input_dim, output_dim).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=10
)


# ==================================================
# 8. Helper functions
# ==================================================
def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)

    return total_loss / total_samples


def get_predictions(model, loader, thresholds):
    """
    thresholds: array-like of shape (n_labels,)
    """
    model.eval()
    all_probs = []
    all_true = []

    thresholds = np.asarray(thresholds).reshape(1, -1)

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)

            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_true.append(yb.numpy())

    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_true)
    y_pred = (y_prob >= thresholds).astype(np.float32)

    return y_true, y_prob, y_pred


def tune_thresholds(model, loader):
    """
    Tune one threshold per label using validation data.
    """
    model.eval()

    all_probs = []
    all_true = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_true.append(yb.numpy())

    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_true)

    n_labels = y_true.shape[1]
    best_thresholds = np.zeros(n_labels)

    for j in range(n_labels):
        best_f1 = -1.0
        best_t = 0.5

        for t in np.linspace(0.1, 0.9, 81):
            pred_j = (y_prob[:, j] >= t).astype(np.float32)
            f1_j = f1_score(y_true[:, j], pred_j, zero_division=0)

            if f1_j > best_f1:
                best_f1 = f1_j
                best_t = t

        best_thresholds[j] = best_t

    return best_thresholds


# ==================================================
# 9. Training with early stopping
# ==================================================
num_epochs = 300
patience = 30

train_losses = []
val_losses = []
val_macro_f1_history = []

best_model_state = None
best_val_macro_f1 = -1.0
epochs_without_improvement = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    num_samples = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        num_samples += xb.size(0)

    train_loss = running_loss / num_samples
    val_loss = evaluate_loss(model, val_loader, criterion)

    # Use temporary 0.5 threshold for model selection during training
    y_val_true, y_val_prob, y_val_pred = get_predictions(model, val_loader, thresholds=[0.5, 0.5])
    val_macro_f1 = f1_score(y_val_true, y_val_pred, average="macro", zero_division=0)

    scheduler.step(val_macro_f1)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_macro_f1_history.append(val_macro_f1)

    if val_macro_f1 > best_val_macro_f1:
        best_val_macro_f1 = val_macro_f1
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Restore best model
model.load_state_dict(best_model_state)


# ==================================================
# 10. Tune thresholds on validation set
# ==================================================
best_thresholds = tune_thresholds(model, val_loader)
print("Best thresholds per label:", best_thresholds)


# ==================================================
# 11. Final evaluation on test set
# ==================================================
y_true, y_prob, y_pred = get_predictions(model, test_loader, thresholds=best_thresholds)

# Elementwise accuracy (same as your previous metric)
elementwise_accuracy = (y_pred == y_true).mean()

# Exact-match accuracy: both labels must be correct for a sample
exact_match_accuracy = np.all(y_pred == y_true, axis=1).mean()

print(f"Test elementwise accuracy: {elementwise_accuracy:.4f}")
print(f"Test exact-match accuracy: {exact_match_accuracy:.4f}")
print(f"Macro F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
print(f"Micro F1: {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
print(f"Per-label F1: {f1_score(y_true, y_pred, average=None, zero_division=0)}")


# ==================================================
# 12. Plots
# ==================================================
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(val_macro_f1_history, label="Validation macro F1")
plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.legend()
plt.tight_layout()
plt.show()