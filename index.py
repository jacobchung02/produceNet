import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import matplotlib.pyplot as plt
from PIL import Image

# Ensure all images are RGB
def convert_to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data paths
data_dir = "data"
train_dir = f"{data_dir}/train"
val_dir = f"{data_dir}/validation"

# Data transforms
transform = transforms.Compose([
    transforms.Lambda(convert_to_rgb),  # Ensure all images are RGB
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model setup
model = models.resnet18(weights='IMAGENET1K_V1')
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=5):
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 20)

        # Training phase
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)

        print(f"Train Loss: {epoch_loss:.4f}")
        print(f"Time taken for epoch: {time.time() - start_time:.2f} seconds")

        # Validation phase
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_loss)

        print(f"Validation Loss: {epoch_loss:.4f}")

    return history

# Benchmark function
def benchmark(model, train_loader, val_loader):
    print("Running benchmarks...")

    # Single-threaded CPU
    model.cpu()
    start = time.time()
    for inputs, labels in train_loader:
        outputs = model(inputs)
    print(f"CPU (Single-threaded) Inference Time: {time.time() - start:.2f} seconds")

    # Parallelized GPU
    model.cuda()
    start = time.time()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
    print(f"GPU (Parallelized) Inference Time: {time.time() - start:.2f} seconds")

# Train the model
history = train_model(model, criterion, optimizer, train_loader, val_loader, epochs=5)

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Run benchmarks
benchmark(model, train_loader, val_loader)

# Save the model
torch.save(model.state_dict(), "model.pth")
print("Model saved!")
