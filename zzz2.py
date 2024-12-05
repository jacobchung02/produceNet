import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Step 1: Load Kaggle Dataset

def load_kaggle_datasets(batch_size=256):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    train_dataset = datasets.ImageFolder(root='./data/train', transform=transform)
    validation_dataset = datasets.ImageFolder(root='./data/validation', transform=transform)
    test_dataset = datasets.ImageFolder(root='./data/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader


# Step 2: Define a Convolutional Neural Network
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
def train_model(model, train_loader, validation_loader, device, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    model.to(device)

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

        print(
            f"Epoch {epoch + 1}/{epochs}, Validation Loss: {validation_loss / len(validation_loader):.4f}, Validation Accuracy: {100 * correct / total:.2f}%")

    end_time = time.time()
    return end_time - start_time


# Step 4: Test Function
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

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


import torch.cuda
import platform

# Print Hardware Details
print("CPU Device Name:", platform.processor())
print("CUDA Available:", torch.cuda.is_available())
print("GPU Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

cpu_device = torch.device('cpu')
gpu_device = torch.device('cuda') if torch.cuda.is_available() else None

# Step 5: Benchmarking
# Load Dataset
batch_size = 256
train_loader, validation_loader, test_loader = load_kaggle_datasets(batch_size=batch_size)

# Define Model
num_classes = len(datasets.ImageFolder(root='./data/train').classes)
nn_model_cpu = create_convnet(3, num_classes)
nn_model_gpu = create_convnet(3, num_classes)

# Single-threaded CPU
torch.set_num_threads(1)
time_cpu = train_model(nn_model_cpu, train_loader, validation_loader, device=cpu_device, epochs=10)
print(f"Single-threaded CPU Training Time: {time_cpu:.2f} seconds")

# Multi-threaded GPU
if gpu_device:
    time_gpu = train_model(nn_model_gpu, train_loader, validation_loader, device=gpu_device, epochs=10)
    print(f"Multi-threaded GPU Training Time: {time_gpu:.2f} seconds")
    test_model(nn_model_gpu, test_loader, device=gpu_device)
else:
    print("GPU not available for benchmarking.")
