import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import resnet18
from model import SSLModel

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define fine-tuning transformations
transform = T.Compose([
    T.Resize(32),
    T.CenterCrop(32),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Load CIFAR-10 dataset for fine-tuning
train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Load the pre-trained model
print("Loading pre-trained model...")
model = SSLModel(resnet18(pretrained=False)).to(device)

# Load the checkpoint from self-supervised learning
checkpoint = torch.load("models/tuned/run1/fine_tuned_checkpoint_epoch_3.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")

# Replace projection head with a classification head
model.projection_head = nn.Sequential(
    nn.Linear(model.projection_head[0].in_features, 10)  # Replace with CIFAR-10 (10 classes)
).to(device)

# Define optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Define loss function (cross-entropy for classification)
criterion = nn.CrossEntropyLoss()

# Fine-tuning loop
print("Starting fine-tuning...")
model.train()
for epoch in range(10):  # Fine-tune for 10 epochs
    epoch_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/10", unit="batch")

    for batch in progress_bar:
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)

        # Forward pass
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update loss and accuracy
        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100. * correct / total:.2f}%")

    scheduler.step()
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {100. * correct / total:.2f}%")

    # Save fine-tuned checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, f"fine_tuned_checkpoint_epoch_{epoch + 1}.pth")
    print(f"Model saved to fine_tuned_checkpoint_epoch_{epoch + 1}.pth")

# Evaluate the fine-tuned model
print("Evaluating fine-tuned model...")
model.eval()
test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Testing", unit="batch"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {100. * correct / total:.2f}%")