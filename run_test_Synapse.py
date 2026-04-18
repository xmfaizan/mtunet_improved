#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
run_test_Synapse.py — Standalone test script for Synapse dataset.
Outputs: per-class DSC, HD95, IoU, and confusion matrix.
Usage:
    python run_test_Synapse.py --checkpoint ./checkpoint/Synapse/mtunet/<best>.pth
"""

import argparse
import logging
import numpy as np
import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import zoom
from medpy import metric
import h5py

from model.MTUNet import MTUNet
from dataset.dataset_Synapse import Synapsedataset

# ── Class names for Synapse ───────────────────────────────────
CLASS_NAMES = [
    'Background', 'Aorta', 'Gallbladder', 'Kidney(L)',
    'Kidney(R)', 'Liver', 'Pancreas', 'Spleen', 'Stomach'
]

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",   required=True,                              help="Path to .pth checkpoint")
parser.add_argument("--root_dir",     default="./MT_UNet_Data/Synapse/train_npz")
parser.add_argument("--list_dir",     default="./dataset/lists_Synapse")
parser.add_argument("--volume_path",  default="./MT_UNet_Data/Synapse/test_vol_h5")
parser.add_argument("--num_classes",  default=9,   type=int)
parser.add_argument("--img_size",     default=224, type=int)
parser.add_argument("--z_spacing",    default=10,  type=int)
parser.add_argument("--output_dir",   default="./test_results/Synapse")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')


# ── Load model ───────────────────────────────────────────────
model = MTUNet(args.num_classes).cuda()
model.load_state_dict(torch.load(args.checkpoint))
model.eval()
logging.info(f"Loaded checkpoint: {args.checkpoint}")


# ── Load test data ───────────────────────────────────────────
db_test = Synapsedataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
testloader = DataLoader(db_test, batch_size=1, shuffle=False)
logging.info(f"Test cases: {len(testloader)}")


# ── Inference ────────────────────────────────────────────────
all_preds   = []
all_targets = []

dice_per_class = np.zeros(args.num_classes - 1)
hd95_per_class = np.zeros(args.num_classes - 1)
iou_per_class  = np.zeros(args.num_classes - 1)
case_count     = 0

with torch.no_grad():
    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
        image = sampled_batch["image"]
        label = sampled_batch["label"]
        case_name = sampled_batch["case_name"][0]

        image_np = image.squeeze(0).cpu().numpy()
        label_np = label.squeeze(0).cpu().numpy()

        # Handle 3D volumes (slice by slice)
        if len(image_np.shape) == 3:
            prediction = np.zeros_like(label_np)
            for ind in range(image_np.shape[0]):
                sl = image_np[ind, :, :]
                x, y = sl.shape
                if x != args.img_size or y != args.img_size:
                    sl = zoom(sl, (args.img_size / x, args.img_size / y), order=3)
                inp = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0).float().cuda()
                out = torch.argmax(torch.softmax(model(inp), dim=1), dim=1).squeeze(0)
                out = out.cpu().numpy()
                if x != args.img_size or y != args.img_size:
                    out = zoom(out, (x / args.img_size, y / args.img_size), order=0)
                prediction[ind] = out
        else:
            inp = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).float().cuda()
            out = torch.argmax(torch.softmax(model(inp), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().numpy()

        # Collect for confusion matrix
        all_preds.extend(prediction.flatten().astype(int).tolist())
        all_targets.extend(label_np.flatten().astype(int).tolist())

        # Per-class metrics
        for c in range(1, args.num_classes):
            pred_c = (prediction == c).astype(int)
            gt_c   = (label_np   == c).astype(int)
            if pred_c.sum() > 0 and gt_c.sum() > 0:
                dice_per_class[c-1] += metric.binary.dc(pred_c, gt_c)
                hd95_per_class[c-1] += metric.binary.hd95(pred_c, gt_c)
                intersection = (pred_c & gt_c).sum()
                union        = (pred_c | gt_c).sum()
                iou_per_class[c-1]  += intersection / (union + 1e-5)
            elif pred_c.sum() == 0 and gt_c.sum() == 0:
                dice_per_class[c-1] += 1.0
                iou_per_class[c-1]  += 1.0

        case_count += 1
        logging.info(f"Case {case_name} done.")

# ── Average metrics ───────────────────────────────────────────
dice_per_class /= case_count
hd95_per_class /= case_count
iou_per_class  /= case_count

mean_dsc  = np.mean(dice_per_class)
mean_hd95 = np.mean(hd95_per_class)
mean_iou  = np.mean(iou_per_class)


# ── Print results table ───────────────────────────────────────
print("\n" + "="*65)
print(f"{'SYNAPSE TEST RESULTS':^65}")
print("="*65)
print(f"{'Class':<20} {'DSC (%)':>10} {'HD95 (mm)':>12} {'IoU (%)':>10}")
print("-"*65)
for i, name in enumerate(CLASS_NAMES[1:]):
    print(f"{name:<20} {dice_per_class[i]*100:>10.2f} {hd95_per_class[i]:>12.2f} {iou_per_class[i]*100:>10.2f}")
print("-"*65)
print(f"{'MEAN':<20} {mean_dsc*100:>10.2f} {mean_hd95:>12.2f} {mean_iou*100:>10.2f}")
print("="*65)

# Save to txt
results_path = os.path.join(args.output_dir, "results.txt")
with open(results_path, "w") as f:
    f.write("SYNAPSE TEST RESULTS\n")
    f.write(f"Checkpoint: {args.checkpoint}\n\n")
    f.write(f"{'Class':<20} {'DSC (%)':>10} {'HD95 (mm)':>12} {'IoU (%)':>10}\n")
    f.write("-"*55 + "\n")
    for i, name in enumerate(CLASS_NAMES[1:]):
        f.write(f"{name:<20} {dice_per_class[i]*100:>10.2f} {hd95_per_class[i]:>12.2f} {iou_per_class[i]*100:>10.2f}\n")
    f.write("-"*55 + "\n")
    f.write(f"{'MEAN':<20} {mean_dsc*100:>10.2f} {mean_hd95:>12.2f} {mean_iou*100:>10.2f}\n")
logging.info(f"Results saved to {results_path}")


# ── Confusion Matrix ──────────────────────────────────────────
all_preds   = np.array(all_preds)
all_targets = np.array(all_targets)

conf_matrix = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
for t, p in zip(all_targets, all_preds):
    if 0 <= t < args.num_classes and 0 <= p < args.num_classes:
        conf_matrix[t, p] += 1

conf_matrix_norm = conf_matrix.astype(float)
row_sums = conf_matrix_norm.sum(axis=1, keepdims=True)
conf_matrix_norm = np.divide(conf_matrix_norm, row_sums,
                             where=row_sums != 0, out=np.zeros_like(conf_matrix_norm))

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Ground Truth', fontsize=12)
ax.set_title('Synapse Confusion Matrix (Normalized)', fontsize=13)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
logging.info(f"Confusion matrix saved to {cm_path}")


# ── IoU Bar Chart ─────────────────────────────────────────────
colors = ['#4C8BE3','#E36B4C','#4CE38B','#E3C24C',
          '#9B4CE3','#E34C9B','#4CE3D9','#E3874C']
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(CLASS_NAMES[1:], iou_per_class * 100, color=colors)
ax.set_ylabel('IoU (%)', fontsize=12)
ax.set_title('Synapse — IoU per Class', fontsize=13)
ax.set_ylim(0, 100)
plt.xticks(rotation=30, ha='right')
for bar, val in zip(bars, iou_per_class * 100):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
iou_path = os.path.join(args.output_dir, "iou_per_class.png")
plt.savefig(iou_path, dpi=150)
plt.close()
logging.info(f"IoU chart saved to {iou_path}")

print(f"\nAll outputs saved to: {args.output_dir}/")