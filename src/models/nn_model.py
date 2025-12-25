import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
DATA_PATH = r"e:\bioimpedance-carcinogenic-tissue\data\data.csv"
MODEL_DIR = r"e:\bioimpedance-carcinogenic-tissue\saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Dataset Class
class BioimpedanceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2. Neural Network Model
class BioImpedanceNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BioImpedanceNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output(x)
        return x

def train_nn_model():
    # Load Data
    df = pd.read_csv(DATA_PATH)
    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    # Preprocessing
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, os.path.join(MODEL_DIR, 'nn_label_encoder.joblib'))

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'nn_scaler.joblib'))

    # Datasets & Loaders
    train_dataset = BioimpedanceDataset(X_train_scaled, y_train)
    test_dataset = BioimpedanceDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize Model
    input_dim = X.shape[1]
    output_dim = len(np.unique(y_encoded))
    model = BioImpedanceNN(input_dim, output_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    epochs = 100
    train_losses = []
    
    print("Starting NN Training...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
        train_losses.append(epoch_loss / len(train_loader))
        
        if (epoch + 1) % 10 == 0:
            acc = 100 * correct / total
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Train Acc: {acc:.2f}%')

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    # Save Model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'nn_model.pth'))
    print("NN Model saved.")

if __name__ == "__main__":
    train_nn_model()
