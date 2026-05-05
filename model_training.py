import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


df = pd.read_csv("Data/Data_processed.csv")
col = df.columns.tolist()

# Reproducibility
torch.manual_seed(676767)

#data preparation
X_df = df.drop(columns=['Have you ever had suicidal thoughts ?', 'Depression'])
y_df = df.drop(columns=['Gender', 'Age', 'Academic Pressure', 'Study Satisfaction', 'Sleep Duration', 'Dietary Habits', 'Study Hours', 'Financial Stress', 'Family History of Mental Illness'])

X = X_df.to_numpy(dtype=np.float32)
y = y_df.to_numpy(dtype=np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train).astype(np.float32)
# X_test = scaler.transform(X_test).astype(np.float32)
#normalization - not performed unless improvment is needed

X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)

X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

#defining model

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

hidden_units_1 = 64
hidden_units_2 = 32

model = nn.Sequential(
    nn.Linear(input_dim, hidden_units_1),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_units_1, hidden_units_2),
    nn.ReLU(),
    nn.Linear(hidden_units_2,output_dim),
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# --------------------------------------------------
# 8. Train
# --------------------------------------------------
num_epochs = 5000
training_error = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    num_samples = 0

    for xb, yb in train_loader:
        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * xb.size(0)
        num_samples += xb.size(0)

    epoch_loss /= num_samples

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:2d}/{num_epochs}, loss={epoch_loss:.4f}")

    training_error.append(epoch_loss)

# --------------------------------------------------
# 9. Evaluate
# --------------------------------------------------
model.eval()
correct = 0
total = 0
all_preds = []
all_true = []


with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        correct += (preds == yb).sum().item()
        total += yb.numel()
        all_preds.append(preds.cpu().numpy())
        all_true.append(yb.cpu().numpy())

y_pred = np.vstack(all_preds)
y_true = np.vstack(all_true)

accuracy = correct / total
print("Test accuracy:", accuracy)

print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
print("Micro F1:", f1_score(y_true, y_pred, average="micro"))
print("Per-label F1:", f1_score(y_true, y_pred, average=None))

plt.plot(training_error)
plt.title("Loss function during training")
plt.xlabel("Number of epochs")
plt.ylabel("Loss function")
plt.show()


torch.save(model.state_dict(), "trained_model.pth")