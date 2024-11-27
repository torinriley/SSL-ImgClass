import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import resnet18


# Define SSLModel with ResNet-18 backbone
class SSLModel(nn.Module):
    def __init__(self, backbone, projection_dim=128):
        super(SSLModel, self).__init__()
        self.backbone = backbone
        self.projection_head = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        self.backbone.fc = nn.Identity()  # Remove classification head

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return projections


# Contrastive Loss
def contrastive_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]

    # Concatenate both views
    z = torch.cat([z_i, z_j], dim=0)  # Shape: (2 * batch_size, projection_dim)

    # Similarity matrix computation (dot product normalized by temperature)
    sim_matrix = torch.mm(z, z.T) / temperature  # Shape: (2 * batch_size, 2 * batch_size)

    # Normalize to prevent instability
    sim_matrix = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0]

    # Mask out self-similarity
    mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, -float("inf"))

    # Extract positive similarities (z_i, z_j) and (z_j, z_i)
    pos_sim = torch.cat([
        torch.diag(sim_matrix, sim_matrix.size(0) // 2),
        torch.diag(sim_matrix, -sim_matrix.size(0) // 2)
    ])

    # Compute contrastive loss
    loss = -torch.log(torch.exp(pos_sim) / torch.sum(torch.exp(sim_matrix), dim=1))
    return loss.mean()


def train_ssl():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Transformations
    transform = T.Compose([
        T.RandomResizedCrop(32),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=3),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Load Dataset
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=4)

    # Initialize Model
    model = SSLModel(resnet18(pretrained=False)).to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Resume Training (if checkpoint exists)
    start_epoch = 1
    checkpoint_path = "models/saves/run2/ssl_checkpoint_epoch_11.pth"
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    # Create "checkpoints" directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)

    # Training Loop
    model.train()
    total_epochs = 15  # Adjust based on the training plan
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", unit="batch")

        for batch in progress_bar:
            imgs, _ = batch
            imgs = imgs.to(device, non_blocking=True)

            # Create two augmented views
            z_i = model(imgs)
            z_j = model(imgs)

            # Validate embeddings
            assert not torch.isnan(z_i).any(), "z_i contains NaN values!"
            assert not torch.isnan(z_j).any(), "z_j contains NaN values!"

            try:
                loss = contrastive_loss(z_i, z_j)
            except Exception as e:
                print(f"Loss computation failed: {e}")
                continue

            optimizer.zero_grad()
            loss.backward()

            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate epoch loss
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        print(f"Epoch {epoch}, Average Loss: {epoch_loss / len(train_loader):.4f}")

        # Save checkpoint
        save_path = f"checkpoints/ssl_checkpoint_epoch_{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train_ssl()

