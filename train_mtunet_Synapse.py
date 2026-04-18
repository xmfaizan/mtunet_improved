#!/usr/bin/env python
# -*- coding:utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision
from utils.utils import DiceLoss, BoundaryLoss   # IMPROVEMENT 6: import BoundaryLoss
from torch.utils.data import DataLoader
from dataset.dataset_Synapse import Synapsedataset, RandomGenerator
import argparse
from tqdm import tqdm
import os
from torchvision import transforms
from utils.test_Synapse import inference
from model.MTUNet import MTUNet
import numpy as np
from medpy.metric import dc, hd95

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=12, type=int, help="batch size")  # fix: type=int
parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
parser.add_argument("--max_epochs", default=100, type=int)                    # fix: type=int
parser.add_argument("--img_size", default=224, type=int)                      # fix: type=int
parser.add_argument("--save_path", default="./checkpoint/Synapse/mtunet")
parser.add_argument("--n_gpu", default=1, type=int)                           # fix: type=int
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--list_dir", default="./dataset/lists_Synapse")
parser.add_argument("--root_dir", default="./MT_UNet_Data/Synapse/train_npz")
parser.add_argument("--volume_path", default="./MT_UNet_Data/Synapse/test_vol_h5")
parser.add_argument("--z_spacing", default=10, type=int)
parser.add_argument("--num_classes", default=9, type=int)                     # fix: type=int
parser.add_argument('--test_save_dir', default='./predictions', help='saving prediction as nii!')
parser.add_argument("--patches_size", default=16, type=int)
parser.add_argument("--n_skip", default=1, type=int)                          # fix: was "--n-skip" (hyphen bug!)
args = parser.parse_args()

model = MTUNet(args.num_classes)  # 9 classes for Synapse

if args.checkpoint:
    model.load_state_dict(torch.load(args.checkpoint))

train_dataset = Synapsedataset(args.root_dir, args.list_dir, split="train", transform=
                               transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
Train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
db_test = Synapsedataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
testloader = DataLoader(db_test, batch_size=1, shuffle=False)

if args.n_gpu > 1:
    model = nn.DataParallel(model)

model = model.cuda()
model.train()

ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(args.num_classes)
boundary_loss = BoundaryLoss(args.num_classes)   # IMPROVEMENT 6: boundary loss instance

save_interval = args.n_skip

iterator = tqdm(range(0, args.max_epochs), ncols=70)
iter_num = 0

Loss = []
Test_Accuracy = []

Best_dcs = 0.7
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s   %(levelname)s   %(message)s')

max_iterations = args.max_epochs * len(Train_loader)
base_lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)

for epoch in iterator:
    model.train()
    train_loss = 0
    for i_batch, sampled_batch in enumerate(Train_loader):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        image_batch = image_batch.type(torch.FloatTensor).cuda()
        label_batch = label_batch.type(torch.FloatTensor).cuda()

        outputs = model(image_batch)

        loss_ce = ce_loss(outputs, label_batch[:].long())
        loss_dice = dice_loss(outputs, label_batch[:], softmax=True)

        # IMPROVEMENT 6: Add boundary loss to training signal
        # Weight breakdown: Dice 40% + CE 40% + Boundary 20%
        # Especially useful for Synapse small organs (pancreas, gallbladder)
        # where boundary precision matters most.
        loss_boundary = boundary_loss(
            torch.softmax(outputs, dim=1), label_batch[:].long(), softmax=False)
        loss = loss_dice * 0.4 + loss_ce * 0.4 + loss_boundary * 0.2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1
        logging.info('iteration %d : loss : %f  dice: %f  ce: %f  boundary: %f  lr: %f' % (
            iter_num, loss.item(), loss_dice.item(),
            loss_ce.item(), loss_boundary.item(), lr_))
        train_loss += loss.item()

    Loss.append(train_loss / len(train_dataset))

    if (epoch + 1) % save_interval == 0:
        avg_dcs, avg_hd = inference(args, model, testloader, args.test_save_dir)

        if avg_dcs > Best_dcs:
            save_mode_path = os.path.join(
                args.save_path,
                'epoch={}_lr={}_avg_dcs={}_avg_hd={}.pth'.format(epoch, lr_, avg_dcs, avg_hd))
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            Best_dcs = avg_dcs

        Test_Accuracy.append(avg_dcs)

    if epoch >= args.max_epochs - 1:
        save_mode_path = os.path.join(
            args.save_path,
            'epoch={}_lr={}_avg_dcs={}.pth'.format(epoch, lr_, avg_dcs))
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
        iterator.close()
        break