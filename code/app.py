import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
from torchvision import models, transforms
from PIL import Image
import io

# Define your neural network for audio processing
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

@st.cache(allow_output_mutation=True)
def load_image_model():
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    model.load_state_dict(torch.load('Models/checkpoint_epoch_9.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache(allow_output_mutation=True)
def load_audio_model():
    model = AudioClassifier()
    model.load_state_dict(torch.load('Audio/best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(model, image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted

def predict_audio(model, audio_path):
    features = extract_features(audio_path)
    features = torch.tensor(features).float().unsqueeze(0)
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

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

def main():
    st.title('Fake Image and Audio Classifier')
    file_uploader = st.file_uploader("Upload a media file", type=['png', 'jpg', 'jpeg', 'wav', 'mp3'])
    if file_uploader is not None:
        file_type = file_uploader.type
        print(file_type)
        with st.spinner('Processing...'):
            if file_type in ['image/jpeg', 'image/png', 'image/jpg']:
                image = Image.open(file_uploader)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                model = load_image_model()
                prediction = predict_image(model, image)
                st.success("Prediction completed.")
                st.write("Prediction:", "FAKE" if prediction.item() == 0 else "REAL")
            elif file_type in ['audio/x-wav','audio/wav', 'audio/mpeg']:
                audio_file = file_uploader.name
                with open(audio_file, 'wb') as f:
                    f.write(file_uploader.getbuffer())
                st.audio(audio_file)
                model = load_audio_model()
                prediction = predict_audio(model, audio_file)
                st.success("Prediction completed.")
                st.write("Prediction:", "FAKE" if prediction == 0 else "REAL")
            else:
                st.error("Unsupported file type. Please upload an image or audio file.")

if __name__ == '__main__':
    main()
