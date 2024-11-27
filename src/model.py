import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import resnet18


class SSLModel(nn.Module):
    def __init__(self, backbone, projection_dim=128):
        super(SSLModel, self).__init__()
        self.backbone = backbone
        self.projection_head = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        self.backbone.fc = nn.Identity()  

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return projections


def contrastive_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]

    # Concatenate both views
    z = torch.cat([z_i, z_j], dim=0)  # (2 * batch_size, projection_dim)

    # Similarity matrix computation (dot product normalized by temperature)
    sim_matrix = torch.mm(z, z.T) / temperature  # (2 * batch_size, 2 * batch_size)

    sim_matrix -= torch.max(sim_matrix, dim=1, keepdim=True)[0]

    # Mask out self-similarity
    mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, -float("inf"))

    # Extract positive similarities (z_i, z_j) and (z_j, z_i)
    pos_sim = torch.cat([
        torch.diag(sim_matrix, sim_matrix.size(0) // 2),
        torch.diag(sim_matrix, -sim_matrix.size(0) // 2)
    ])

    loss = -torch.log(torch.exp(pos_sim) / torch.sum(torch.exp(sim_matrix), dim=1))
    return loss.mean()


if __name__ == "__main__":
    transform = T.Compose([
        T.RandomResizedCrop(32),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=3),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        pin_memory=True,
        num_workers=4  
    )

    model = SSLModel(resnet18(pretrained=False)).to(device := torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Training loop
    model.train()
    for epoch in range(10):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/10", unit="batch")

        for batch in progress_bar:
            imgs, _ = batch
            imgs = imgs.to(device, non_blocking=True)

            # Create two augmented views
            z_i = model(imgs)
            z_j = model(imgs)

            # Compute contrastive loss
            try:
                loss = contrastive_loss(z_i, z_j)
            except Exception as e:
                print(f"Loss computation failed: {e}")
                continue

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        print(f"Epoch {epoch + 1}, Average Loss: {epoch_loss / len(train_loader):.4f}")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, f"ssl_checkpoint_epoch_{epoch + 1}.pth")
        print(f"Model saved to ssl_checkpoint_epoch_{epoch + 1}.pth")

