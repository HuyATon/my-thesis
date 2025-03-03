import torch
import cv2
import argparse
from network.network_pro import Inpaint
from dataset.inpainting_dataset import InpaintingDataset
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

# Fixed configs (mask folders...)
mask_sub_folders = ["0_2", "2_4", "4_6"]
mask_folders = [
    "/media02/nnthao05/data/mask/test_mask/" + sub for sub in mask_sub_folders
]
print(mask_folders)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
out_dir = {}
# Args
parser = argparse.ArgumentParser(description="Inpaint inference (fixed mask path [0_2_4_6])")
parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--img", type=str, required=True, help="Path to image folder")
parser.add_argument("--output", type=str, required=True, help="Path to output folder")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

args = parser.parse_args()
ckpt_path = args.ckpt
img_path = args.img
batch_size = args.batch_size

for index, mask_type in enumerate(mask_sub_folders):
    out_path = os.path.join(args.output, mask_type)
    os.makedirs(out_path, exist_ok=True)
    out_dir[mask_folders[index]] = out_path

# Load model
model = Inpaint().to(device)
model_checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(model_checkpoint["model_state_dict"])

model.eval()
for mask_folder in mask_folders:
    index = 0
    test_dataset = InpaintingDataset(img_dir=img_path, mask_dir=mask_folder)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    for inputs, targets in test_loader:
        imgs, masks = inputs[0].to(device), inputs[1].to(device)
        targets = targets.to(device)
        outputs = model(imgs, masks)
        
        for i in range(outputs.shape[0]):
            inpainted_img = outputs[i].permute(1, 2, 0).cpu().detach().numpy() * 255.
            inpainted_img = inpainted_img.astype('uint8')
            file_name = str(index).zfill(5) + ".png"
            save_path = os.path.join(out_dir[mask_folder], file_name)
            cv2.imwrite(save_path, inpainted_img)
            index += 1
