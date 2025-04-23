
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv").dropna()

X = df.drop("Diabetes_binary", axis=1).values
y = df["Diabetes_binary"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

class DiabetesModel(nn.Module):
    def __init__(self):
        super(DiabetesModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = DiabetesModel()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0008)
epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

model.eval()
y_true, y_pred, y_scores = [], [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())
        y_scores.extend(outputs.numpy())

import numpy as np
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_scores = np.array(y_scores)

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, digits=4))
print("\n🧱 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print(f"\n🔵 ROC AUC Score: {roc_auc_score(y_true, y_scores):.4f}")
accuracy = (y_pred == y_true).mean()
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
