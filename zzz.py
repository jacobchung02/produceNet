import torch
import torch.nn as nn
import torch.optim as optim
import time

# Step 1: Generate Sample Data
def generate_sample_data(samples=500000, features=100, classes=10):
    X = torch.randn(samples, features)  # Random features
    y = torch.randint(0, classes, (samples,))  # Random class labels
    return X, y

# Step 2: Define a More Complex Neural Network
def create_model(input_size, hidden_size, output_size):
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size * 2),
        nn.ReLU(),
        nn.Linear(hidden_size * 2, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
        nn.Softmax(dim=1)
    )
    return model

# Step 3: Train Function
def train_model(model, X, y, device, epochs=20, batch_size=256):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    model.to(device)
    X, y = X.to(device), y.to(device)

    start_time = time.time()
    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            inputs = X[i:i+batch_size]
            targets = y[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    end_time = time.time()
    return end_time - start_time

import torch.cuda

import platform

# Print Hardware Details
print("CPU Device Name:", platform.processor())
print("CUDA Available:", torch.cuda.is_available())
print("GPU Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


cpu_device = torch.device('cpu')
gpu_device = torch.device('cuda') if torch.cuda.is_available() else None

# Step 4: Benchmarking
# Generate Data
X, y = generate_sample_data()

input_size = 100
hidden_size = 128
output_size = 10

# Single-threaded CPU
torch.set_num_threads(1)
nn_model_cpu = create_model(input_size, hidden_size, output_size)
time_cpu = train_model(nn_model_cpu, X, y, device=cpu_device)
print(f"Single-threaded CPU Training Time: {time_cpu:.2f} seconds")

# Multi-threaded GPU
if gpu_device:
    nn_model_gpu = create_model(input_size, hidden_size, output_size)
    time_gpu = train_model(nn_model_gpu, X, y, device=gpu_device)
    print(f"Multi-threaded GPU Training Time: {time_gpu:.2f} seconds")
else:
    print("GPU not available for benchmarking.")


