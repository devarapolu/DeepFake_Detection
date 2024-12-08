import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
import tqdm

# Dataset loading and transformations
def load_data(data_dir, batch_size):
    data_transforms = {
        'Train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Validation': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['Train', 'Validation']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                   for x in ['Train', 'Validation']}
    return dataloaders

# Modify VGG16 for binary classification
def initialize_model(num_classes):
    model = models.vgg16(pretrained=True) 
    # Freeze training for all "features" layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_inputs, num_classes)  # Modify the last layer
    return model

# Training and validation function
def train_model(model, dataloaders, criterion, optimizer, save_path, num_epochs=25, gamma=0.95):
    best_acc = 0.0
    train_acc, val_acc, train_loss, val_loss = [], [], [], []
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm.tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'Train':
                train_acc.append(epoch_acc.item())
                train_loss.append(epoch_loss)
            else:
                val_acc.append(epoch_acc.item())
                val_loss.append(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    torch.save(model.state_dict(), os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth'))
                    print('Best model saved with accuracy: {:.4f}'.format(best_acc))

        scheduler.step()  # Decay the learning rate

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc, val_acc, train_loss, val_loss
# Plotting the learning curves
def plot_learning_curves(train_acc, val_acc, train_loss, val_loss):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('image_read1.png')
    plt.show()

# Main function to run the setup
if __name__ == '__main__':
    data_dir = 'Image-Dataset'  # Update this path
    save_path = '../Models'  # Update this path
    os.makedirs(save_path, exist_ok=True)
    batch_size = 32
    num_classes = 2
    num_epochs = 10
    learning_rate = 0.001

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloaders = load_data(data_dir, batch_size)
    model = initialize_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model, train_acc, val_acc, train_loss, val_loss = train_model(model, dataloaders, criterion, optimizer, save_path, num_epochs=num_epochs)
    plot_learning_curves(train_acc, val_acc, train_loss, val_loss)
