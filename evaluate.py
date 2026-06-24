import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import UNet
from dataset import ISBIDataset


# ── Cấu hình ──────────────────────────────────────────────────────────────────
TEST_IMG_DIR   = '/kaggle/input/datasets/hamzamohiuddin/isbi-2012-challenge/unmodified-data/test/imgs'
TEST_LABEL_DIR = '/kaggle/input/datasets/hamzamohiuddin/isbi-2012-challenge/unmodified-data/test/labels'
MODEL_PATH     = '/kaggle/input/models/habangchu/unet/pytorch/default/1/unet_isbi_model.pth'
# ──────────────────────────────────────────────────────────────────────────────


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset & DataLoader
    test_dataset = ISBIDataset(TEST_IMG_DIR, TEST_LABEL_DIR)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load model
    model = UNet(n_channels=1, n_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Đã load model thành công!")

    total_iou = 0

    with torch.no_grad():
        for i, (images, masks, _) in enumerate(test_loader):
            images = images.to(device)
            masks  = masks.to(device)

            outputs = model(images)
            preds   = torch.argmax(outputs, dim=1)

            # Cắt mask gốc cho khớp với đầu ra (388×388)
            diff       = (masks.size(-1) - preds.size(-1)) // 2
            true_masks = masks[:, diff:diff + preds.size(-1), diff:diff + preds.size(-1)]

            # Tính IoU
            intersection = (preds * true_masks).sum()
            union        = (preds + true_masks).gt(0).sum()
            iou          = (intersection + 1e-6) / (union + 1e-6)
            total_iou   += iou.item()

            # Visualize 3 ảnh đầu
            if i < 3:
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1); plt.imshow(images[0, 0].cpu(), cmap='gray'); plt.title("Ảnh gốc")
                plt.subplot(1, 3, 2); plt.imshow(true_masks[0].cpu(), cmap='gray'); plt.title("Nhãn chuẩn")
                plt.subplot(1, 3, 3); plt.imshow(preds[0].cpu(), cmap='gray'); plt.title("U-Net dự đoán")
                plt.tight_layout()
                plt.show()

    print(f"Mean IoU trên tập Test: {total_iou / len(test_loader):.4f}")


if __name__ == '__main__':
    main()
