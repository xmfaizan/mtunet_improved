#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import torch
from medpy import metric
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import SimpleITK as sitk
from scipy.ndimage import zoom

class Normalize():
    def __call__(self, sample):
        function = transforms.Normalize((.5 , .5, .5), (0.5, 0.5, 0.5))
        return function(sample[0]), sample[1]


class ToTensor():
    def __call__(self, sample):
        function = transforms.ToTensor()
        return function(sample[0]), function(sample[1])


class RandomRotation():
    def __init__(self):
        pass

    def __call__(self, sample):
        img, label = sample
        random_angle = np.random.randint(0, 360)
        return img.rotate(random_angle, Image.NEAREST), label.rotate(random_angle, Image.NEAREST)


class RandomFlip():
    def __init__(self):
        pass

    def __call__(self, sample):
        img, label = sample
        temp = np.random.random()
        if temp > 0 and temp < 0.25:
            return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)
        elif temp >= 0.25 and temp < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM), label.transpose(Image.FLIP_TOP_BOTTOM)
        elif temp >= 0.5 and temp < 0.75:
            return img.transpose(Image.ROTATE_90), label.transpose(Image.ROTATE_90)
        else:
            return img, label


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


# ============================================================
# IMPROVEMENT 6: Boundary Loss
# Penalizes errors near organ boundaries more heavily.
# Works by computing the distance transform of the ground
# truth boundary and weighting the loss by it — so pixels
# near edges contribute more to the total loss.
# ============================================================
class BoundaryLoss(nn.Module):
    def __init__(self, n_classes):
        super(BoundaryLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _get_boundary(self, mask):
        """Extract boundary pixels from a binary mask using max pooling trick."""
        # mask: (B, 1, H, W) float tensor
        pad = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        erode = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        boundary = pad - erode  # pixels that are on the edge
        return boundary

    def forward(self, inputs, target, softmax=False):
        """
        inputs: (B, n_classes, H, W) raw logits or softmax probs
        target: (B, H, W) integer class labels
        """
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        target_one_hot = self._one_hot_encoder(target)  # (B, n_classes, H, W)

        loss = 0.0
        for i in range(1, self.n_classes):  # skip background class 0
            pred_i = inputs[:, i:i+1, :, :]        # (B, 1, H, W)
            target_i = target_one_hot[:, i:i+1, :, :]  # (B, 1, H, W)

            # Get boundary region of ground truth
            boundary = self._get_boundary(target_i)  # (B, 1, H, W)

            # Boundary-weighted BCE: penalize errors at boundaries more
            boundary_weight = 1.0 + 5.0 * boundary  # boundary pixels get 6x weight
            bce = F.binary_cross_entropy(pred_i, target_i, weight=boundary_weight, reduction='mean')
            loss += bce

        return loss / (self.n_classes - 1)


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    return metric_list