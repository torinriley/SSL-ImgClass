import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from model import SSLModel  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading fine-tuned model...")
model = SSLModel(resnet18(pretrained=False)).to(device)

model.projection_head = torch.nn.Sequential(
    torch.nn.Linear(512, 10) 
).to(device)

checkpoint_path = "fine_tuned_checkpoint_epoch_3.pth" 
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Model loaded from {checkpoint_path}")

transform = T.Compose([
    T.Resize(32),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

classes = test_dataset.classes

print("Extracting embeddings and predictions...")
embeddings = []
predictions = []

with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs) 
        _, preds = outputs.max(1) 
        embeddings.append(outputs.cpu().numpy()) 
        predictions.append(preds.cpu().numpy())

embeddings = np.concatenate(embeddings, axis=0)
predictions = np.concatenate(predictions, axis=0)

print("Reducing dimensionality using t-SNE...")
tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
reduced_embeddings = tsne.fit_transform(embeddings)

def plot_clusters(embeddings, labels, class_names, title="t-SNE Visualization of Predictions"):
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7
    )
    legend = plt.legend(
        handles=scatter.legend_elements()[0],
        labels=class_names,
        loc="upper right",
        title="Classes"
    )
    plt.gca().add_artist(legend)
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()

plot_clusters(reduced_embeddings, predictions, classes, title="t-SNE Visualization of Predicted Labels")
