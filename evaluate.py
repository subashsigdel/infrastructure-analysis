"""
evaluate.py

Evaluate the infrastructure segmentation model on a folder of test images.

Expected folder structure:
    /path/to/test_root/
        img1.jpg
        img1_mask.png
        img2.jpg
        img2_mask.png
        ...

Usage:
    python evaluate.py --data_root Final/test_extracted \
                       --checkpoint flair_unet_14cls_best.pth \
                       --num_samples 5
"""

import argparse
from pathlib import Path
import random

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# -----------------------------
# Hard-coded class list
# -----------------------------
CLASS_NAMES = [
    "background",          # 0
    "Bridge",              # 1
    "Building",            # 2
    "Cottage",             # 3
    "Dam",                 # 4
    "Haystack",            # 5
    "House",               # 6
    "Irrigation Channel",  # 7
    "Road",                # 8
    "Temple",              # 9
    "Wall",                # 10
    "log",                 # 11
]

NUM_CLASSES = len(CLASS_NAMES)
TILE_SIZE = 1024

# FLAIR normalization stats (0–255 space)
FLAIR_MEANS = np.array([105.08, 110.87, 101.82], dtype=np.float32)
FLAIR_STDS  = np.array([52.17,  45.38,  44.00], dtype=np.float32)


# -----------------------------
# Dataset for test folder
# -----------------------------
class TestFolderDataset(Dataset):
    """
    Dataset that scans a folder for *.jpg and expects masks as <stem>_mask.png.
    Images and masks are resized to TILE_SIZE x TILE_SIZE.
    """

    def __init__(self, root_dir: Path, tile_size: int = TILE_SIZE):
        self.root_dir = Path(root_dir)
        self.tile_size = tile_size
        self.samples = []

        jpgs = sorted(self.root_dir.rglob("*.jpg"))
        if not jpgs:
            raise RuntimeError(f"No .jpg files found in {self.root_dir}")

        for img_path in jpgs:
            mask_name = f"{img_path.stem}_mask.png"
            mask_path = img_path.with_name(mask_name)
            if not mask_path.exists():
                print(f"[WARN] Mask not found for {img_path.name} (expected {mask_name}), skipping.")
                continue
            self.samples.append((img_path, mask_path))

        if not self.samples:
            raise RuntimeError("No (image, mask) pairs found in test folder.")

        print(f"[INFO] Found {len(self.samples)} (image, mask) pairs in {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Load image (RGB)
        img_pil = Image.open(str(img_path)).convert("RGB")
        img_np = np.array(img_pil).astype(np.float32)  # [H,W,3], 0–255

        # Load mask
        mask_pil = Image.open(str(mask_path))
        mask_np = np.array(mask_pil)

        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

        # Resize to TILE_SIZE x TILE_SIZE
        if img_np.shape[0] != self.tile_size or img_np.shape[1] != self.tile_size:
            img_pil = img_pil.resize((self.tile_size, self.tile_size), Image.BILINEAR)
            img_np = np.array(img_pil).astype(np.float32)

        if mask_np.shape[0] != self.tile_size or mask_np.shape[1] != self.tile_size:
            mask_pil = mask_pil.resize((self.tile_size, self.tile_size), Image.NEAREST)
            mask_np = np.array(mask_pil)

        # FLAIR normalization
        img_np = (img_np - FLAIR_MEANS) / FLAIR_STDS

        # To tensors
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()  # [3,H,W]
        mask_tensor = torch.from_numpy(mask_np).long()                  # [H,W]

        return img_tensor, mask_tensor, str(img_path.name)


# -----------------------------
# Model loading
# -----------------------------
def build_model(num_classes: int, checkpoint: Path, device: torch.device):
    """Create UNet(ResNet34) and load trained weights, PyTorch 2.6-safe."""
    # Load checkpoint (handle weights_only in PyTorch 2.6)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)

    # If it's a full model
    if isinstance(state, torch.nn.Module):
        model = state.to(device)
        model.eval()
        return model

    # Otherwise assume state_dict-like
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model


# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    """
    Compute:
      - confusion matrix
      - per-class IoU
      - per-class precision, recall, F1
      - mean IoU
      - macro-averaged precision, recall, F1
      - pixel accuracy
      - per-class support (GT pixels)
    """
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch_idx, (images, masks, _) in enumerate(loader, start=1):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)              # [B, C, H, W]
        preds = torch.argmax(logits, 1)     # [B, H, W]

        gt = masks.cpu().numpy().reshape(-1)
        pr = preds.cpu().numpy().reshape(-1)

        # valid labels
        valid = (gt >= 0) & (gt < num_classes)
        gt = gt[valid]
        pr = pr[valid]

        k = num_classes * gt + pr
        hist = np.bincount(k, minlength=num_classes ** 2)
        conf += hist.reshape(num_classes, num_classes)

        if batch_idx % 10 == 0 or batch_idx == len(loader):
            print(f"[EVAL] Processed {batch_idx}/{len(loader)} batches")

    # ---- per-class stats ----
    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp   # predicted as c but GT != c
    fn = conf.sum(axis=1) - tp   # GT is c but predicted != c
    tn = conf.sum() - (tp + fp + fn)

    # avoid division by zero
    eps = 1e-7

    # IoU
    denom_iou = tp + fp + fn + eps
    iou = tp / denom_iou

    # precision, recall, F1
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    # supports
    support = conf.sum(axis=1)  # GT pixels per class

    # overall metrics
    pixel_acc = float(tp.sum() / conf.sum())
    miou      = float(np.mean(iou))
    macro_p   = float(np.mean(precision))
    macro_r   = float(np.mean(recall))
    macro_f1  = float(np.mean(f1))

    # ---- pretty print ----
    print("\n===== GLOBAL METRICS =====")
    print(f"Overall pixel accuracy : {pixel_acc:.4f}")
    print(f"Mean IoU               : {miou:.4f}")
    print(f"Macro Precision        : {macro_p:.4f}")
    print(f"Macro Recall           : {macro_r:.4f}")
    print(f"Macro F1               : {macro_f1:.4f}")
    print()

    print("===== PER-CLASS METRICS =====")
    header = (
        "ID  Class                "
        "Support     IoU     Prec    Rec     F1"
    )
    print(header)
    print("-" * len(header))

    for i in range(num_classes):
        name = CLASS_NAMES[i]
        print(
            f"{i:2d}  {name:20s} "
            f"{int(support[i]):8d}  "
            f"{iou[i]:6.3f}  "
            f"{precision[i]:6.3f}  "
            f"{recall[i]:6.3f}  "
            f"{f1[i]:6.3f}"
        )

    return {
        "confusion_matrix": conf,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "pixel_accuracy": pixel_acc,
        "mean_iou": miou,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
    }



# -----------------------------
# Visualization utilities
# -----------------------------
def unnormalize_image(image_tensor):
    """
    image_tensor: [3, H, W], normalized with FLAIR stats (0–255).
    returns: [H, W, 3] in [0,1]
    """
    img = image_tensor.permute(1, 2, 0).cpu().numpy()  # [H,W,3]
    img = img * FLAIR_STDS + FLAIR_MEANS
    img = np.clip(img / 255.0, 0, 1)
    return img


def mask_to_color(mask, num_classes):
    """
    mask: [H, W] int class ids
    returns: [H, W, 3] float32, in [0,1]
    """
    cmap = plt.get_cmap("tab20", num_classes)
    colored = cmap(mask % num_classes)[..., :3]
    return colored


@torch.no_grad()
def save_eval_samples(model, dataset, device, out_dir: Path, num_samples: int = 3):
    """
    Save a few evaluation samples with:
        - original image (unnormalized)
        - GT mask (colored)
        - Pred mask (colored)
        - legend showing class names
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(dataset)
    indices = random.sample(range(n), k=min(num_samples, n))

    print(f"[INFO] Saving {len(indices)} evaluation samples to {out_dir}")

    for idx in indices:
        img_tensor, mask_tensor, img_name = dataset[idx]
        img_tensor = img_tensor.to(device).unsqueeze(0)  # [1,3,H,W]
        mask_np = mask_tensor.numpy()

        logits = model(img_tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Unnormalize image for display
        img_rgb = unnormalize_image(img_tensor.squeeze(0).cpu())  # [H,W,3]

        gt_color = mask_to_color(mask_np, NUM_CLASSES)
        pred_color = mask_to_color(pred, NUM_CLASSES)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(img_rgb)
        axs[0].set_title(f"Image: {img_name}")
        axs[0].axis("off")

        axs[1].imshow(gt_color)
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")

        axs[2].imshow(pred_color)
        axs[2].set_title("Prediction")
        axs[2].axis("off")

        # Legend
        handles = []
        cmap = plt.get_cmap("tab20", NUM_CLASSES)
        for i, name in enumerate(CLASS_NAMES):
            color = cmap(i)
            label = f"{i}: {name}"
            handles.append(mpatches.Patch(color=color, label=label))

        fig.legend(handles=handles, loc="center right",
                   bbox_to_anchor=(1.15, 0.5), title="Classes")
        fig.tight_layout(rect=[0, 0, 0.85, 1])

        out_path = out_dir / f"{Path(img_name).stem}_eval_sample.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[INFO] Saved sample: {out_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model on test folder.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to folder with test images + *_mask.png.")
    parser.add_argument("--checkpoint", type=str, default="flair_unet_14cls_best.pth",
                        help="Path to trained model checkpoint (.pth).")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of DataLoader workers.")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of qualitative samples to save.")
    parser.add_argument("--samples_out_dir", type=str, default="eval_samples",
                        help="Directory to save qualitative sample images.")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    checkpoint = Path(args.checkpoint)
    samples_out_dir = Path(args.samples_out_dir)

    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Build dataset + loader
    dataset = TestFolderDataset(data_root)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers)

    # Build model
    model = build_model(NUM_CLASSES, checkpoint, device)

    # Evaluate
    metrics = evaluate(model, loader, device, NUM_CLASSES)

    # (Optional) Save metrics to JSON for slides/report
    import json
    with open("eval_metrics.json", "w") as f:
        json.dump(
            {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in metrics.items()},
            f,
            indent=2
        )
    print("[INFO] Saved metrics to eval_metrics.json")

    # Save qualitative samples
    save_eval_samples(model, dataset, device, samples_out_dir, num_samples=args.num_samples)



if __name__ == "__main__":
    main()
