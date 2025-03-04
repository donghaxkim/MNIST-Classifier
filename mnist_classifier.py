import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Defining device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
])

# Download and load the training data
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Check dataset size
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")


# Define a simple feedfoward neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128) # Input --> Hidden
        self.fc2= nn.Linear(128, 64) # Hidden --> Hidden2
        self.fc3= nn.Linear(64, 10) # Hidden2 --> output

    def forward(self, x):
        x = torch.flatten(x, start_dim=1) # flatten 28x28 image to 784 long vector
        x = torch.relu(self.fc1(x)) # ReLU activation fc1
        x = torch.relu(self.fc2(x)) # ReLU activation fc2
        x = self.fc3(x) # no activation for output
        return x
    
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
net = Net().to(device)  # Create an instance of the model
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

print("Loss function and optimizer initialized.")

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

print("Training complete!")

torch.save(net.state_dict(), "mnist_model.pth")
print("Model saved!")

net = Net()
net.load_state_dict(torch.load("mnist_model.pth"))
net.eval()  # Set the model to evaluation mode


import matplotlib.pyplot as plt

dataiter = iter(test_loader)
images, labels = next(dataiter)

outputs = net(images.to(device))
_, predicted = torch.max(outputs, 1)

fig = plt.figure(figsize=(12, 6))

for idx in range(5):
    ax = fig.add_subplot(1, 5, idx + 1)
    ax.imshow(images[idx].cpu().numpy().squeeze(), cmap="gray")
    ax.set_title(f"Pred: {predicted[idx].item()}", fontsize=14, color='royalblue', fontweight='bold')
    ax.axis('off')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

fig.patch.set_facecolor('lightgray')

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

fig.suptitle('MNIST Test Image Predictions', fontsize=18, fontweight='bold', color='darkred')

plt.show()

