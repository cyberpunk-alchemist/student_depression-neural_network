import torch
import torch.nn as nn
import numpy as np

input_dim = 9      # must match training
output_dim = 2     # your two target labels

hidden_units_1 = 64
hidden_units_2 = 32

model = nn.Sequential(
    nn.Linear(input_dim, hidden_units_1),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_units_1, hidden_units_2),
    nn.ReLU(),
    nn.Linear(hidden_units_2, output_dim),
)

model.load_state_dict(torch.load("trained_model.pth"))
model.eval()

#['Gender', 'Age', 'Academic Pressure', 'Study Satisfaction', 'Sleep Duration', 'Dietary Habits', 'Study Hours', 'Financial Stress', 'Family History of Mental Illness']
x_new = np.array([[1, 21, 5, 3, 8, 0.8, 6, 1, 1]], dtype=np.float32) 
x_tensor = torch.tensor(x_new)

with torch.no_grad():
    logits = model(x_tensor)
    probs = torch.sigmoid(logits)

print("Probabilities:", probs.numpy())
print("[suicidal thoughts, depression]")