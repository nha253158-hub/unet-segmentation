import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from scipy.ndimage import distance_transform_edt, label


def get_weight_map(mask_binary, w0=10, sigma=5):
    """
    Tính weight map theo công thức bài báo U-Net.
    mask_binary: nhãn dạng 0-1 (nền và tế bào)
    """
    mask_labeled, n_labels = label(mask_binary)

    if n_labels <= 1:
        return np.ones(mask_binary.shape)

    wc = np.ones(mask_binary.shape)
    cells = np.unique(mask_labeled)[1:]  # Bỏ nền (0)

    # Tính bản đồ khoảng cách cho từng tế bào
    dist_maps = np.stack([distance_transform_edt(mask_labeled != cell_id) for cell_id in cells])
    dist_maps = np.sort(dist_maps, axis=0)

    # d1: khoảng cách tới tế bào gần nhất, d2: gần thứ hai
    d1 = dist_maps[0]
    d2 = dist_maps[1]

    # Công thức bài báo
    w = wc + w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2))

    # Chỉ áp dụng trọng số cao cho vùng nền để phân tách tế bào
    w[mask_binary > 0] = 1.0
    return w


class ISBIDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_names = sorted(os.listdir(img_dir))
        self.mask_names = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])

        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        image = np.array(image).astype(np.float32) / 255.0
        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_labels = (mask_np > 0.5).astype(np.int64)

        # Tính Weight Map
        w_map = get_weight_map(mask_labels)

        # Pad ảnh đúng chuẩn bài báo
        image = np.pad(image, ((30, 30), (30, 30)), mode='reflect')

        # Chuyển sang Tensor
        image = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_labels)
        weight_tensor = torch.from_numpy(w_map).float()

        return image, mask_tensor, weight_tensor
