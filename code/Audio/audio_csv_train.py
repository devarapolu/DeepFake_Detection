import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Dataset class that handles CSV data
class AudioFeaturesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label

# Load data and create dataloaders
def load_data(csv_file, batch_size):
    data = pd.read_csv(csv_file)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data.iloc[:, -1])
    features = data.iloc[:, :-1].values.astype(np.float32)

    # Split the dataset
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42)

    # Create Dataset instances
    train_dataset = AudioFeaturesDataset(train_features, train_labels)
    test_dataset = AudioFeaturesDataset(test_features, test_labels)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train and evaluate the model
def train_model(model, train_loader, test_loader, epochs, device, save_path,gamma):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    best_acc = 0.0
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        test_losses.append(epoch_loss)
        test_accuracies.append(epoch_acc)

        # Checkpointing
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            print(f'Checkpoint saved at epoch {epoch + 1} with validation accuracy of {best_acc:.4f}')

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.4f}')

    return train_losses, test_losses, train_accuracies, test_accuracies

# Plotting the learning curves
def plot_learning_curves(train_losses, test_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('audio_read.png')
    plt.show()

# Main function to execute the training process
if __name__ == '__main__':
    csv_file = '/Audio-dataset/KAGGLE/DATASET-balanced.csv'  # Update this path
    save_path = '/Audio'  # Update this path
    os.makedirs(save_path, exist_ok=True)
    batch_size = 32
    input_size = 26  # Update this if your feature size differs
    epochs = 55
    gamma = 0.95
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = load_data(csv_file, batch_size)
    model = SimpleNN(input_size).to(device)
    train_losses, test_losses, train_accuracies, test_accuracies = train_model(model, train_loader, test_loader, epochs, device, save_path,gamma)
    plot_learning_curves(train_losses, test_losses, train_accuracies, test_accuracies)
