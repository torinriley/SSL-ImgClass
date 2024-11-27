import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from model import SSLModel  

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved SSL model
model = SSLModel(resnet18(pretrained=False)).to(device)
saved_model_path = "models/saves/run2/ssl_checkpoint_epoch_12.pth"
checkpoint = torch.load(saved_model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Model loaded from {saved_model_path}")

transform = T.Compose([
    T.Resize(32),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

# Extract embeddings and corresponding labels
embeddings = []
labels = []

print("Extracting embeddings...")
with torch.no_grad():
    for imgs, lbls in dataloader:
        imgs = imgs.to(device)
        z = model(imgs)  # Get the embeddings
        embeddings.append(z.cpu().numpy())
        labels.append(lbls.numpy())

# Concatenate all embeddings and labels
embeddings = np.concatenate(embeddings, axis=0)
labels = np.concatenate(labels, axis=0)

# Reduce dimensionality using t-SNE
print("Reducing dimensionality...")
tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
reduced_embeddings = tsne.fit_transform(embeddings)

# Plot embeddings
def plot_embeddings(embeddings, labels, class_names):
    plt.figure(figsize=(10, 8))
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
    plt.title("t-SNE Visualization of SSL Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()

# Get CIFAR-10 class names
class_names = dataset.classes

# Plot the embeddings
plot_embeddings(reduced_embeddings, labels, class_names)

