#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
xai_visualization.py
--------------------
Explainable AI for MT-UNet+.
Generates for each test case:
  1. GradCAM heatmap overlay on input image
  2. Attention map from the MTM bottleneck transformer block
  3. Side-by-side panel: Input | GT | Prediction | GradCAM | Attention Map

Usage:
    python xai_visualization.py --dataset ACDC --checkpoint ./checkpoint/ACDC/mtunet/<best>.pth
    python xai_visualization.py --dataset Synapse --checkpoint ./checkpoint/Synapse/mtunet/<best>.pth
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader
from scipy.ndimage import zoom
from tqdm import tqdm
import cv2

from model.MTUNet import MTUNet
from dataset.dataset_ACDC import ACDCdataset
from dataset.dataset_Synapse import Synapsedataset

# ─────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",      required=True,  choices=["ACDC", "Synapse"])
parser.add_argument("--checkpoint",   required=True,  help="Path to best .pth checkpoint")
parser.add_argument("--img_size",     default=224,    type=int)
parser.add_argument("--num_cases",    default=5,      type=int,  help="How many test cases to visualize")
parser.add_argument("--slice_idx",    default=None,   type=int,  help="Specific slice index (None = middle slice)")
parser.add_argument("--output_dir",   default=None,   help="Output folder (auto-set if None)")

# Dataset-specific defaults
parser.add_argument("--acdc_root",      default="./MT_UNet_Data/ACDC")
parser.add_argument("--acdc_list",      default="./dataset/lists_ACDC")
parser.add_argument("--acdc_test",      default="./MT_UNet_Data/ACDC/test")
parser.add_argument("--synapse_root",   default="./MT_UNet_Data/Synapse/train_npz")
parser.add_argument("--synapse_list",   default="./dataset/lists_Synapse")
parser.add_argument("--synapse_test",   default="./MT_UNet_Data/Synapse/test_vol_h5")
args = parser.parse_args()

# Auto-set output dir
if args.output_dir is None:
    args.output_dir = f"./xai_results/{args.dataset}"
os.makedirs(args.output_dir, exist_ok=True)

# Dataset config
if args.dataset == "ACDC":
    NUM_CLASSES  = 4
    CLASS_NAMES  = ['Background', 'RV', 'Myocardium', 'LV']
    CLASS_COLORS = np.array([[0,0,0], [255,0,0], [0,255,0], [0,0,255]], dtype=np.uint8)
else:
    NUM_CLASSES  = 9
    CLASS_NAMES  = ['BG','Aorta','Gallbladder','Kidney(L)','Kidney(R)',
                    'Liver','Pancreas','Spleen','Stomach']
    CLASS_COLORS = np.array([
        [0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,0],
        [255,0,255],[0,255,255],[128,0,255],[255,128,0]], dtype=np.uint8)

if args.dataset == "ACDC":
    args.num_classes = NUM_CLASSES


# ─────────────────────────────────────────────────────────────
# GRADCAM IMPLEMENTATION
# Hooks into a target conv layer and computes the weighted
# activation map using gradients from the output prediction.
# ─────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        """
        model:        the MTUNet model
        target_layer: nn.Module — the layer to hook into
                      (we use the last ConvBNReLU in the ResNet50 stem)
        """
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class):
        """
        input_tensor:  (1, C, H, W) on CUDA
        target_class:  int — the organ class to explain
        Returns:
            cam: (H, W) numpy array, values in [0, 1]
        """
        self.model.zero_grad()
        output = self.model(input_tensor)          # (1, num_classes, H, W)

        # Score for target class: sum of predicted probabilities for that class
        score = output[0, target_class, :, :].sum()
        score.backward(retain_graph=True)

        # Global average pooling of gradients over spatial dims
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam


# ─────────────────────────────────────────────────────────────
# ATTENTION MAP EXTRACTOR
# Extracts attention weights from the MTM bottleneck block.
# The External Attention (EA) module's linear_0 layer produces
# attention scores — we visualize their spatial average.
# ─────────────────────────────────────────────────────────────
class AttentionExtractor:
    def __init__(self, model):
        self.model       = model
        self.attn_maps   = []
        self._register_hooks()

    def _register_hooks(self):
        """
        Hook into the EA module's linear_0 layer inside bottleneck.
        This captures the attention score matrix before softmax.
        """
        def hook_fn(module, input, output):
            # output: (B, N, heads, k) attention logits
            attn = output.detach().cpu()
            # Average across heads and k dims → (B, N)
            attn_avg = attn.abs().mean(dim=-1).mean(dim=-1)  # (B, N)
            self.attn_maps.append(attn_avg)

        # Hook into both EAmodule blocks in the bottleneck
        for module in self.model.bottleneck.modules():
            if hasattr(module, 'linear_0'):
                module.linear_0.register_forward_hook(hook_fn)

    def get_attention_map(self, h, w):
        """
        Returns the spatial attention map resized to (h, w).
        Call this after a forward pass.
        """
        if not self.attn_maps:
            return np.zeros((h, w))

        # Average all collected attention maps
        attn = torch.stack(self.attn_maps, dim=0).mean(dim=0)  # (B, N)
        attn = attn[0]  # take first batch: (N,)

        # Reshape to spatial grid
        n    = attn.shape[0]
        side = int(np.sqrt(n))
        if side * side != n:
            return np.zeros((h, w))

        attn_2d = attn.numpy().reshape(side, side)

        # Normalize
        if attn_2d.max() > attn_2d.min():
            attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min())

        # Resize to image size
        attn_resized = cv2.resize(attn_2d, (w, h), interpolation=cv2.INTER_LINEAR)
        self.attn_maps = []  # clear for next forward pass
        return attn_resized


# ─────────────────────────────────────────────────────────────
# HELPER: colorize segmentation mask
# ─────────────────────────────────────────────────────────────
def colorize_mask(mask, colors):
    h, w   = mask.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for c_idx, color in enumerate(colors):
        canvas[mask == c_idx] = color
    return canvas


# ─────────────────────────────────────────────────────────────
# HELPER: overlay heatmap on grayscale image
# ─────────────────────────────────────────────────────────────
def overlay_heatmap(gray_img, heatmap, alpha=0.5):
    """
    gray_img: (H, W) float in [0,1]
    heatmap:  (H, W) float in [0,1]
    Returns:  (H, W, 3) uint8 RGB
    """
    # Normalize gray to [0,255]
    gray_rgb = np.stack([gray_img * 255]*3, axis=-1).astype(np.uint8)

    # Apply jet colormap to heatmap
    heat_colored = (cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)

    # Blend
    blended = (alpha * heat_colored + (1 - alpha) * gray_rgb).astype(np.uint8)
    return blended


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
print(f"\nLoading model from: {args.checkpoint}")
model = MTUNet(NUM_CLASSES).cuda()
model.load_state_dict(torch.load(args.checkpoint))
model.eval()

# Set up GradCAM on the ResNet50 encoder's projection conv
# (last conv layer before features go into MTM)
target_layer = model.stem.model.proj[0]  # Conv2d(512, 256, 1)
gradcam      = GradCAM(model, target_layer)
attn_extractor = AttentionExtractor(model)

print(f"GradCAM target layer: {target_layer}")
print(f"Output dir: {args.output_dir}\n")


# ─────────────────────────────────────────────────────────────
# LOAD TEST DATA
# ─────────────────────────────────────────────────────────────
if args.dataset == "ACDC":
    db_test = ACDCdataset(
        base_dir=args.acdc_test,
        list_dir=args.acdc_list,
        split="test")
else:
    db_test = Synapsedataset(
        base_dir=args.synapse_test,
        list_dir=args.synapse_list,
        split="test")

testloader = DataLoader(db_test, batch_size=1, shuffle=False)
num_cases  = min(args.num_cases, len(testloader))
print(f"Visualizing {num_cases} cases from {args.dataset} test set...\n")


# ─────────────────────────────────────────────────────────────
# MAIN VISUALIZATION LOOP
# ─────────────────────────────────────────────────────────────
for i_batch, sampled_batch in tqdm(enumerate(testloader), total=num_cases):
    if i_batch >= num_cases:
        break

    case_name = sampled_batch["case_name"][0]
    image_np  = sampled_batch["image"].squeeze(0).cpu().numpy()
    label_np  = sampled_batch["label"].squeeze(0).cpu().numpy()

    # ── Pick the slice to visualize ──────────────────────────
    if len(image_np.shape) == 3:
        # 3D volume: pick middle slice or user-specified
        if args.slice_idx is not None:
            sl_idx = min(args.slice_idx, image_np.shape[0] - 1)
        else:
            sl_idx = image_np.shape[0] // 2
        slice_img = image_np[sl_idx, :, :]
        slice_lbl = label_np[sl_idx, :, :]
    else:
        slice_img = image_np
        slice_lbl = label_np
        sl_idx    = 0

    h_orig, w_orig = slice_img.shape

    # ── Resize for model input ───────────────────────────────
    if h_orig != args.img_size or w_orig != args.img_size:
        slice_resized = zoom(slice_img, (args.img_size/h_orig, args.img_size/w_orig), order=3)
    else:
        slice_resized = slice_img

    # Normalize slice to [0,1] for display
    s_min, s_max = slice_resized.min(), slice_resized.max()
    if s_max > s_min:
        slice_norm = (slice_resized - s_min) / (s_max - s_min)
    else:
        slice_norm = slice_resized

    inp = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().cuda()

    # ── Standard forward pass for prediction ─────────────────
    with torch.no_grad():
        output  = model(inp)
        pred    = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
        pred_np = pred.cpu().numpy()

    # Resize prediction back to original size
    if h_orig != args.img_size or w_orig != args.img_size:
        pred_np = zoom(pred_np, (h_orig/args.img_size, w_orig/args.img_size), order=0)

    # ── GradCAM: pick the most predicted non-background class ─
    unique_classes = np.unique(pred_np)
    unique_classes = unique_classes[unique_classes > 0]  # skip background
    target_cls     = int(unique_classes[0]) if len(unique_classes) > 0 else 1

    # Need gradients for GradCAM — re-run with grad enabled
    model.zero_grad()
    inp_grad = inp.clone().requires_grad_(True)
    cam_map  = gradcam.generate(inp_grad, target_class=target_cls)

    # Get attention map (collected during GradCAM forward pass)
    attn_map = attn_extractor.get_attention_map(args.img_size, args.img_size)

    # Resize CAM and attention to original size
    cam_resized  = cv2.resize(cam_map,  (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    attn_resized = cv2.resize(attn_map, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    # ── Colorize masks ────────────────────────────────────────
    slice_display = (slice_norm * 255).astype(np.uint8)  # for display
    slice_norm_f  = slice_display.astype(float) / 255.0

    gt_color   = colorize_mask(slice_lbl.astype(int),  CLASS_COLORS)
    pred_color = colorize_mask(pred_np.astype(int),    CLASS_COLORS)

    gradcam_overlay = overlay_heatmap(slice_norm_f, cam_resized,  alpha=0.55)
    attn_overlay    = overlay_heatmap(slice_norm_f, attn_resized, alpha=0.55)

    # ── Build 5-panel figure ──────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(
        f'{args.dataset} | Case: {case_name} | Slice: {sl_idx} | '
        f'GradCAM class: {CLASS_NAMES[target_cls]}',
        fontsize=11, fontweight='bold')

    titles = ['Input Image', 'Ground Truth', 'Prediction',
              f'GradCAM\n({CLASS_NAMES[target_cls]})', 'Attention Map']
    images = [
        np.stack([slice_display]*3, axis=-1),
        gt_color,
        pred_color,
        gradcam_overlay,
        attn_overlay
    ]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(
        args.output_dir,
        f'{case_name}_slice{sl_idx}_xai.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# SUMMARY GRID: all cases in one figure
# ─────────────────────────────────────────────────────────────
print("\nGenerating summary grid...")

case_images = []
for fname in sorted(os.listdir(args.output_dir)):
    if fname.endswith('_xai.png'):
        img = plt.imread(os.path.join(args.output_dir, fname))
        case_images.append(img)

if case_images:
    fig, axes = plt.subplots(len(case_images), 1,
                             figsize=(22, 4.5 * len(case_images)))
    if len(case_images) == 1:
        axes = [axes]
    for ax, img in zip(axes, case_images):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    grid_path = os.path.join(args.output_dir, f'{args.dataset}_XAI_summary.png')
    plt.savefig(grid_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Summary grid saved: {grid_path}")

print(f"\nDone. All XAI outputs in: {args.output_dir}/")