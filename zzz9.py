import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# uncomment out if you have intel cpu
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Avoid OpenMP library duplication

# Step 1: Load Kaggle Dataset
# Function to load the Kaggle dataset with specified batch size.

def load_kaggle_datasets(batch_size):
    # Use faster interpolation and resize
    transform = transforms.Compose([
        transforms.Resize((128, 128), interpolation=InterpolationMode.BILINEAR),  # Resize images
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Load datasets using ImageFolder
    train_dataset = datasets.ImageFolder(root='./data/train', transform=transform)
    validation_dataset = datasets.ImageFolder(root='./data/validation', transform=transform)
    test_dataset = datasets.ImageFolder(root='./data/test', transform=transform)

    # Define optimal DataLoader settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, validation_loader, test_loader


# Step 2: Define a Convolutional Neural Network
# Function to create a convolutional neural network (CNN) with specified input channels and output classes.
def create_convnet(input_channels, num_classes):
    # Define the layers of the CNN.
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
    # Define the loss function and optimizer.
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


# Step 5: Plot History
# Function to plot the training and validation accuracy for multiple models.
def plot_batch_histories(histories, labels, batch_size):
    plt.figure(figsize=(12, 6))

    for history, label in zip(histories, labels):
        if f'Batch {batch_size}' in label:
            plt.plot(history['validation_accuracy'], label=label)

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Validation Accuracy for Batch Size {batch_size}')
    plt.legend()
    plt.show()


def plot_training_times(times, labels, batch_size):
    plt.figure(figsize=(10, 6))

    # Assign colors: blue for CPU, red for GPU
    colors = ['blue' if 'CPU' in label else 'red' for label in labels]

    plt.bar(labels, times, color=colors)
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.title(f'Training Time for Batch Size {batch_size}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


import torch.cuda
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")
import platform

# Print Hardware Details
print("CPU Device Name:", platform.processor())
print("CUDA Available:", torch.cuda.is_available())
print("GPU Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# Define the CPU and GPU devices.
cpu_device = torch.device('cpu')
gpu_device = torch.device('cuda') if torch.cuda.is_available() else None

# Step 6: Benchmarking
batch_sizes = [128, 256]
cpu_threads = [1, 2, 4, torch.get_num_threads()]
histories = []
labels = []
training_times = []

for batch_size in batch_sizes:
    print(f"\nTraining with Batch Size: {batch_size}\n")
    train_loader, validation_loader, test_loader = load_kaggle_datasets(batch_size=batch_size)

    # CPU Models
    for threads in cpu_threads:
        torch.set_num_threads(threads)
        print(f"\nTraining CPU Model with {threads} Thread(s):")
        nn_model_cpu = create_convnet(3, len(datasets.ImageFolder(root='./data/train').classes))
        time_cpu, history = train_model(nn_model_cpu, train_loader, validation_loader, device=cpu_device, epochs=10)
        test_accuracy_cpu = test_model(nn_model_cpu, test_loader, device=cpu_device)
        print(f"CPU Training Time ({threads} Threads): {time_cpu:.2f} seconds, Test Accuracy: {test_accuracy_cpu:.2f}%")

        histories.append(history)
        training_times.append(time_cpu)
        labels.append(f'CPU {threads} Threads (Batch {batch_size})')

    # GPU Model
    print(f"\nTraining GPU Model:")
    nn_model_gpu = create_convnet(3, len(datasets.ImageFolder(root='./data/train').classes))
    time_gpu, history = train_model(nn_model_gpu, train_loader, validation_loader, device=gpu_device, epochs=10)
    test_accuracy_gpu = test_model(nn_model_gpu, test_loader, device=gpu_device)
    print(f"GPU Training Time: {time_gpu:.2f} seconds, Test Accuracy: {test_accuracy_gpu:.2f}%")

    histories.append(history)
    training_times.append(time_gpu)
    labels.append(f'GPU (Batch {batch_size})')

    if batch_size == 128:
        plot_training_times(training_times[:len(cpu_threads) + 1], labels[:len(cpu_threads) + 1], batch_size)
    else:
        plot_training_times(training_times[len(cpu_threads) + 1:], labels[len(cpu_threads) + 1:], batch_size)

# Plot Histories
plot_batch_histories(histories, labels, 128)
plot_batch_histories(histories, labels, 256)
