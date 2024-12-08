import torch
import librosa
import numpy as np
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.fc1 = nn.Linear(26, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Assuming binary classification: FAKE or REAL

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model(model_path):
    model = AudioClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    features = np.array([
        np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        np.mean(librosa.feature.rms(y=y)),
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        np.mean(librosa.feature.zero_crossing_rate(y))
    ])
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)
    features = np.hstack((features, mfccs_mean))
    return features

def predict(model, features):
    features = torch.tensor(features).float().unsqueeze(0)
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

def process_folder(base_folder_path, model):
    predictions = []
    labels = []
    label_encoder = LabelEncoder()

    # Loop through each label folder (FAKE and REAL)
    for label_folder in os.listdir(base_folder_path):
        folder_path = os.path.join(base_folder_path, label_folder)
        if os.path.isdir(folder_path):
            file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]

            for file_path in file_paths:

                features = extract_features(file_path)
                prediction = predict(model, features)
                predictions.append(prediction)
                labels.append(label_folder)

    labels_encoded = label_encoder.fit_transform(labels)
    cm = confusion_matrix(labels_encoded, predictions)
    return cm, label_encoder.classes_

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    model_path = "best_model.pth"  # Update this path
    audio_base_folder_path = "Audio-dataset/KAGGLE/AUDIO"  # Update this path, should contain 'FAKE' and 'REAL' subfolders

    model = load_model(model_path)
    cm, classes = process_folder(audio_base_folder_path, model)
    print("Confusion Matrix:")
    print(cm)
    print("Class Labels:", classes)
    plot_confusion_matrix(cm, classes)

