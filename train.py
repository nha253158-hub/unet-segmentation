import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import UNet
from dataset import ISBIDataset


# ── Cấu hình ──────────────────────────────────────────────────────────────────
IMG_DIR  = '/kaggle/input/datasets/hamzamohiuddin/isbi-2012-challenge/unmodified-data/train/imgs'
MASK_DIR = '/kaggle/input/datasets/hamzamohiuddin/isbi-2012-challenge/unmodified-data/train/labels'

EPOCHS     = 100
BATCH_SIZE = 8
LR         = 1e-3
# ──────────────────────────────────────────────────────────────────────────────


def weighted_cross_entropy(logits, targets, weights):
    loss = F.cross_entropy(logits, targets, reduction='none')
    return (loss * weights).mean()


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset & DataLoader
    dataset = ISBIDataset(IMG_DIR, MASK_DIR)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model = UNet(n_channels=1, n_classes=2).to(device)
    model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    print("Bắt đầu huấn luyện...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for images, masks, weight_maps in train_loader:
            images      = images.to(device)
            masks       = masks.long().to(device)
            weight_maps = weight_maps.to(device)

            preds = model(images)

            # Cắt masks/weights cho khớp với output
            diff     = (masks.size(-1) - preds.size(-1)) // 2
            masks_v  = masks[:, diff:-diff, diff:-diff]
            weights_v = weight_maps[:, diff:-diff, diff:-diff]

            loss = weighted_cross_entropy(preds, masks_v, weights_v)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] — Loss: {epoch_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), 'unet_isbi_model.pth')
    print("Huấn luyện hoàn tất! Model lưu tại unet_isbi_model.pth")


if __name__ == '__main__':
    main()
