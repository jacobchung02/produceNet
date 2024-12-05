import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Step 1: Load Kaggle Dataset
# Function to load the Kaggle dataset with specified batch size.
def load_kaggle_datasets(batch_size):
    # Define transformations for image preprocessing.
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Load training dataset.
    train_dataset = datasets.ImageFolder(root='./data/train', transform=transform)
    # Load validation dataset.
    validation_dataset = datasets.ImageFolder(root='./data/validation', transform=transform)
    # Load testing dataset.
    test_dataset = datasets.ImageFolder(root='./data/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader


# Function to check if all images and their specific classes are loaded correctly.
def check_dataset_images(dataset, dataset_name):
    # Dictionary to count images per class
    class_counts = {class_name: 0 for class_name in dataset.classes}

    # Iterate through the dataset
    for image, label in dataset:
        class_name = dataset.classes[label]
        class_counts[class_name] += 1

    # Print the results
    print(f"\n{dataset_name} Class Counts:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")


# Step 2: Define a Convolutional Neural Network
# Function to create a convolutional neural network (CNN) with specified input channels and output classes.
def create_convnet(input_channels, num_classes):
    layers = [
        nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 32 * 32, 128),  # Adjust for image size
        nn.ReLU(),
        nn.Linear(128, num_classes)
    ]
    return nn.Sequential(*layers)


# Step 3: Train Function
# Function to train the model on the training dataset and validate on the validation dataset.
def train_model(model, train_loader, validation_loader, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    model.to(device)

    history = {'validation_loss': [], 'validation_accuracy': []}

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation Loop
        model.eval()
        validation_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                validation_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        history['validation_loss'].append(validation_loss / len(validation_loader))
        history['validation_accuracy'].append(100 * correct / total)

        print(
            f"Epoch {epoch + 1}/{epochs}, Validation Loss: {validation_loss / len(validation_loader):.4f}, Validation Accuracy: {100 * correct / total:.2f}%")

    end_time = time.time()
    return end_time - start_time, history


# Step 4: Test Function
# Function to test the model on the test dataset and calculate accuracy.
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


import torch.cuda
import platform

# Print Hardware Details
print("CPU Device Name:", platform.processor())
print("CUDA Available:", torch.cuda.is_available())
print("GPU Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# Define the CPU and GPU devices.
cpu_device = torch.device('cpu')
gpu_device = torch.device('cuda') if torch.cuda.is_available() else None

# Load the datasets
batch_size = 128
train_loader, validation_loader, test_loader = load_kaggle_datasets(batch_size=batch_size)

# Verify images and classes for each dataset
check_dataset_images(train_loader.dataset, "Training Dataset")
check_dataset_images(validation_loader.dataset, "Validation Dataset")
check_dataset_images(test_loader.dataset, "Test Dataset")
