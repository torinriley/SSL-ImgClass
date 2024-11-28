import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from model import SSLModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.Resize(32),
    T.CenterCrop(32),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print("Loading fine-tuned model...")
model = SSLModel(resnet18(pretrained=False)).to(device)

model.projection_head = torch.nn.Sequential(
    torch.nn.Linear(512, 10) 
).to(device)

checkpoint_path = "models/tuned/run1/fine_tuned_checkpoint_epoch_3.pth"  
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Model loaded from {checkpoint_path}")

criterion = torch.nn.CrossEntropyLoss()
total_loss = 0
correct = 0
total = 0

print("Evaluating the fine-tuned model...")
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

accuracy = 100. * correct / total
avg_loss = total_loss / len(test_loader)

print(f"Validation Results: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
