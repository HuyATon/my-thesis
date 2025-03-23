import torch
import cv2
import argparse
from network.network_pro import Inpaint
from dataset.inpainting_dataset import InpaintingDataset
from torch.utils.data import DataLoader
import os
import numpy as np

mask_sub_folders = ["0_2"]
mask_folders = [
    "/content/drive/MyDrive/KLTN/DATA/testing_mask_dataset/" + sub for sub in mask_sub_folders
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Argument Parser
parser = argparse.ArgumentParser(description="Inpaint inference (fixed mask path [0_2_4_6])")
parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--img", type=str, required=True, help="Path to image folder")
parser.add_argument("--output", type=str, required=True, help="Path to output folder")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

args = parser.parse_args()

# Validate paths
if not os.path.exists(args.ckpt):
    raise FileNotFoundError(f"Checkpoint file '{args.ckpt}' not found.")
if not os.path.exists(args.img):
    raise FileNotFoundError(f"Image folder '{args.img}' not found.")
for mask_folder in mask_folders:
    if not os.path.exists(mask_folder):
        raise FileNotFoundError(f"Mask folder '{mask_folder}' not found.")

# Create output directories
os.makedirs(args.output, exist_ok=True)
out_dir = {}
for index, mask_type in enumerate(mask_sub_folders):
    out_path = os.path.join(args.output, mask_type)
    os.makedirs(out_path, exist_ok=True)
    out_dir[mask_folders[index]] = out_path

# Load model
model = Inpaint().to(device)
checkpoint = torch.load(args.ckpt, map_location=device)

if "model_state_dict" not in checkpoint:
    raise KeyError("Invalid checkpoint: Missing 'model_state_dict'")

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Inference
with torch.no_grad():
    for mask_folder in mask_folders:
        test_dataset = InpaintingDataset(img_dir=args.img, mask_dir=mask_folder)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        for index, (inputs, _) in enumerate(test_loader):  # Starts from 0
            imgs, masks = inputs[0].to(device), inputs[1].to(device)
            _, _, img_h, img_w = imgs.shape
            masks = torch.nn.functional.interpolate(masks, size=(img_h, img_w), mode="nearest")

            outputs = model(imgs, masks)

            for i in range(outputs.shape[0]):
                out = torch.clamp(outputs[i], 0, 1)  # Ensure valid range
                inpainted_img = (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

              
                file_name = f"{index:05d}.png"
                save_path = os.path.join(out_dir[mask_folder], file_name)
                cv2.imwrite(save_path, inpainted_img)
