import os
import cv2
import torch
from torch.utils.data import Dataset

class InpaintingDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        img = cv2.imread(img_path) / 255.0  # Normalize
        mask = cv2.imread(mask_path)[..., 0] / 255.0  

        img = torch.Tensor(img).permute(2, 0, 1).float()
        mask = torch.Tensor(mask).unsqueeze(0).float()

        return (img, mask), img