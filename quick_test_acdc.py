import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from model.MTUNet import MTUNet
from dataset.dataset_ACDC import ACDCdataset, RandomGenerator
from utils.test_ACDC import inference

args = argparse.Namespace(
    num_classes=4,
    img_size=224,
    z_spacing=10,
    root_dir='./MT_UNet_Data/ACDC',
    list_dir='./dataset/lists_ACDC',
    volume_path='./MT_UNet_Data/ACDC/test',
    test_save_dir=None
)

# Load untrained model
model = MTUNet(args.num_classes).cuda()

# Load test data
db_test = ACDCdataset(
    base_dir=args.volume_path,
    list_dir=args.list_dir,
    split="test"
)
testloader = DataLoader(db_test, batch_size=1, shuffle=False)

# Run inference
print("Running test on untrained model (expect low DSC ~0.0-0.2)...")
avg_dcs, avg_hd = inference(args, model, testloader, test_save_path=None)
print(f"DSC: {avg_dcs:.4f}  HD95: {avg_hd:.4f}")