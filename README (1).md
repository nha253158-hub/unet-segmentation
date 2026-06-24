# UNet Image Segmentation — ISBI 2012

Implementation of the original **U-Net** architecture (Ronneberger et al., 2015) for biomedical image segmentation on the **ISBI 2012 EM Segmentation Challenge** dataset.

---

## Overview

| Item | Detail |
|------|--------|
| Dataset | ISBI 2012 (electron microscopy, neuronal structures) |
| Task | Binary segmentation (cell / background) |
| Framework | PyTorch |
| Platform | Kaggle (GPU T4) |

---

## Architecture

Faithful re-implementation of the original U-Net paper:

- **Encoder**: 4 downsampling blocks (double conv, no padding) + MaxPool
- **Bottleneck**: 512 → 1024 channels
- **Decoder**: 4 upsampling blocks (ConvTranspose2d + crop-and-concat skip connections)
- **Output**: 1×1 Conv → 2-class segmentation map

Key details matching the paper:
- No padding in convolutions (valid convolutions)
- Crop-and-concat skip connections to handle size mismatch
- Reflect padding on input images (+30px each side)

---

## Loss Function

Custom **weighted cross-entropy** to handle class imbalance and cell separation:

```
w(x) = wc + w0 * exp(-((d1 + d2)²) / (2σ²))
```

where `d1`, `d2` are distances to the two nearest cell borders (w0=10, σ=5).

---

## Training

```
Optimizer : Adam (lr=1e-3)
Epochs    : 100
Batch size: 8
Init      : He initialization (Kaiming normal)
Grad clip : max_norm=1.0
```

---

## Results

| Metric | Score |
|--------|-------|
| Mean IoU (test set) | **0.8657** |

---

## File Structure

```
unet-segmentation/
└── unet_segmentation.ipynb   # Full pipeline: model, dataset, train, eval
```

---

## References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). MICCAI.
- Dataset: [ISBI 2012 EM Segmentation Challenge](http://brainiac2.mit.edu/isbi_challenge/)
