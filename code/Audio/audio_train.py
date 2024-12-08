import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Dataset class that extracts audio features
class AudioFeaturesDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.labels = {'FAKE': 0, 'REAL': 1}
        self.files = []
        self.load_files()

    def load_files(self):
        for label in ['FAKE', 'REAL']:
            path = os.path.join(self.directory, label)
            files = os.listdir(path)
            for f in files:
                self.files.append((os.path.join(path, f), self.labels[label]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        audio, sr = librosa.load(file_path, sr=None)
        return self.extract_features(audio, sr), label

    def extract_features(self, audio, sr):
        chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr).mean()
        rms = librosa.feature.rms(y=audio).mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr).mean()
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr).mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio).mean()
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs, axis=1)
        features = np.hstack((chroma_stft, rms, spectral_centroid, spectral_bandwidth, rolloff,
                              zero_crossing_rate, mfccs_mean))
        return torch.tensor(features, dtype=torch.float32)

# Neural network model for audio features
class AudioFeaturesNet(nn.Module):
    def __init__(self):
        super(AudioFeaturesNet, self).__init__()
        self.fc1 = nn.Linear(26, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Training and validation function
def train_and_validate_model(directory, epochs=10, batch_size=4, learning_rate=0.001):
    dataset = AudioFeaturesDataset(directory)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = AudioFeaturesNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0
    best_model = None
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch'):
            print(device)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(100 * train_correct / train_total)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(100 * val_correct / val_total)
        val_acc = 100 * val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
            torch.save(best_model, os.path.join(save_path, 'best_model.pth'))  # Save the best model
            print(f'Saved new best model with accuracy: {val_acc}% at epoch {epoch+1}')

    # Save the final model after all epochs
    final_model_path = os.path.join(save_path, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved at {final_model_path}')
    # Plot learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('audio_read.png')
    plt.show()

    return model

# Example usage
if __name__ == '__main__':
    train_and_validate_model('Audio-dataset/KAGGLE/AUDIO', epochs=20, batch_size=32, learning_rate=0.001)

