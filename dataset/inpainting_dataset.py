import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class InpaintingDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.transform_to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        img = self.transform_to_tensor(img) * 2 - 1. # RGB [-1, 1] (CHW: 3 x 256 x 256)

        mask = Image.open(mask_path).convert("L")
        mask = self.transform_to_tensor(mask) # RGB [0, 1] (CHW: 1 x 256 x 256)

        return (img, mask), img