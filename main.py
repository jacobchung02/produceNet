import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import time
import os
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import platform
import warnings

# For Intel CPU users, uncomment to prevent potential OpenMP library duplication error(s)
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Ignore warning when loading images
warnings.filterwarnings("ignore", category = UserWarning, module = "PIL.Image")

# Function preloads datasets based on batch size
def load_kaggle_datasets(batch_size):
    # Transform images into smaller sizes using bilinear interpolation
    transform = transforms.Compose(
        [
        transforms.Resize((128, 128), interpolation = InterpolationMode.BILINEAR),  # Resize images
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ]
    )

    # Load datasets using ImageFolder
    train_dataset = datasets.ImageFolder(root = './data/train', transform = transform)
    validation_dataset = datasets.ImageFolder(root = './data/validation', transform = transform)
    test_dataset = datasets.ImageFolder(root = './data/test', transform = transform)

    # Set DataLoader parameters for each directory
    train_loader = DataLoader(
        train_dataset,              # Dataset corresponds to directory name
        batch_size = batch_size,    # 128 or 256
        shuffle = True              # Randomize contents
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size = batch_size,
        shuffle = False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False
    )

    return train_loader, validation_loader, test_loader


# Constructs CNN with specified input channels and output classes
def create_convnet(input_channels, num_classes):
    layers = [
        # First conv. layer
        nn.Conv2d(input_channels, 32, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2),

        # Second conv. layer
        nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2),

        # Flatten tensor to 1D vector
        nn.Flatten(),

        # Fully-connected layer reduces image size to desired dimensions (128x128)
        nn.Linear(64 * 32 * 32, 128),  
        nn.ReLU(),
        nn.Linear(128, num_classes)
    ]
    return nn.Sequential(*layers)


# Apply training dataset and validation dataset to test target model
def train_model(model, train_loader, validation_loader, device, epochs=10):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    model.to(device)

    # Track history of each epoch within the active benchmark
    history = {'validation_loss': [], 'validation_accuracy': []}

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        # Training loop
        for images, classes in train_loader:
            # Moves inputs to specified device
            images, classes = images.to(device), classes.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, classes)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        validation_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, classes in validation_loader:
                images, classes = images.to(device), classes.to(device)
                outputs = model(images)
                loss = criterion(outputs, classes)
                validation_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += classes.size(0)
                correct += (predicted == classes).sum().item()

        # Append loss and accuracy to history
        history['validation_loss'].append(validation_loss / len(validation_loader))
        history['validation_accuracy'].append(100 * correct / total)

        # Print performance data of each epoch
        print(
            f"Epoch {epoch + 1}/{epochs}, Validation Loss: {validation_loss / len(validation_loader):.4f}, Validation Accuracy: {100 * correct / total:.2f}%")

    end_time = time.time()
    return end_time - start_time, history


# Tests accuracy of the model on the dataset
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    # No gradient; use inferencing
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # Print and return accuracy as % at the end of each epoch
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


# Function plots the training and validation accuracy for each model as a line graph
def plot_batch_histories(histories, labels, batch_size):
    plt.figure(figsize=(12, 6))

    for history, label in zip(histories, labels):
        if f'Batch {batch_size}' in label:
            plt.plot(history['validation_accuracy'], label = label)

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Validation Accuracy for Batch Size {batch_size}')
    plt.legend()
    plt.show()


# Function plots the training time for each model as a bar graph
def plot_training_times(times, labels, batch_size):
    plt.figure(figsize=(10, 6))

    # Assign colors: blue for CPU, red for GPU
    colors = ['blue' if 'CPU' in label else 'red' for label in labels]

    plt.bar(labels, times, color = colors)
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.title(f'Training Time for Batch Size {batch_size}')
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()


# For debugging purposes; prints processor model & whether your machine has CUDA. If True, print the CUDA device model
print("CPU Device Name:", platform.processor())
print("CUDA Available:", torch.cuda.is_available())
print("GPU Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# Set CPU. If CUDA exists on the GPU, set GPU as well
cpu_device = torch.device('cpu')
gpu_device = torch.device('cuda') if torch.cuda.is_available() else None

# Test CPU models based on threads. Initialized at 1, 2, 4, and max threads in CPU. Can be changed based on machine
cpu_threads = [1, 2, 4, torch.get_num_threads()]
# Set batch sizes; two in this case
batch_sizes = [128, 256]

# Initialize arrays to populate with test data for plotting
labels = []
histories = []
training_times = []

# Benchmark each CPU + GPU thread model for each batch size
for batch_size in batch_sizes:
    print(f"\nTraining with Batch Size: {batch_size}\n")
    train_loader, validation_loader, test_loader = load_kaggle_datasets(batch_size=batch_size)

    # CPU models; 1 per thread count
    for threads in cpu_threads:
        torch.set_num_threads(threads)
        print(f"\nTraining CPU Model with {threads} Thread(s):")
        nn_model_cpu = create_convnet(3, len(datasets.ImageFolder(root='./data/train').classes))
        time_cpu, history = train_model(nn_model_cpu, train_loader, validation_loader, device=cpu_device, epochs=10)
        test_accuracy_cpu = test_model(nn_model_cpu, test_loader, device=cpu_device)
        print(f"CPU Training Time ({threads} Threads): {time_cpu:.2f} seconds, Test Accuracy: {test_accuracy_cpu:.2f}%")

        # Map each result from CPU to the graph depending on batch
        histories.append(history)
        training_times.append(time_cpu)
        labels.append(f'CPU {threads} Threads (Batch {batch_size})')

        # Save CPU model to project directory
        torch.save(nn_model_cpu.state_dict(), f"{threads}threadcpu{batch_size}.pth")

    # GPU model
    print(f"\nTraining GPU Model:")
    nn_model_gpu = create_convnet(3, len(datasets.ImageFolder(root='./data/train').classes))
    time_gpu, history = train_model(nn_model_gpu, train_loader, validation_loader, device=gpu_device, epochs=10)
    test_accuracy_gpu = test_model(nn_model_gpu, test_loader, device=gpu_device)
    print(f"GPU Training Time: {time_gpu:.2f} seconds, Test Accuracy: {test_accuracy_gpu:.2f}%")

    # Map each result from GPU to the graph depending on batch
    histories.append(history)
    training_times.append(time_gpu)
    labels.append(f'GPU (Batch {batch_size})')

    # Save GPU model to project directory
    torch.save(nn_model_gpu.state_dict(), f"gpu{batch_size}.pth")

    # Plot first batch onto training time graph, then any subsequent batch gets mapped to its own graph
    if batch_size == 128:
        plot_training_times(training_times[:len(cpu_threads) + 1], labels[:len(cpu_threads) + 1], batch_size)
    else:
        plot_training_times(training_times[len(cpu_threads) + 1:], labels[len(cpu_threads) + 1:], batch_size)

# Plot histories onto line graph once all tests have concluded
plot_batch_histories(histories, labels, 128)
plot_batch_histories(histories, labels, 256)
