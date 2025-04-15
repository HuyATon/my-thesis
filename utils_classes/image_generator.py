import torch
from network.network_pro import Inpaint
from dataset.inpainting_dataset import InpaintingDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
import os

class ImageGenerator:
    def __init__(self, img_dir, mask_dir, output_dir, batch_size=8, model_checkpoint_path = None):
        self.model_checkpoint_path = model_checkpoint_path
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir
        self.batch_size = batch_size

        self.assess_ranges = ["0_2", "2_4", "4_6"]
        self.out_dirs = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Inpaint().to(self.device)
        self.mask_sub_folder_paths = [
            os.path.join(mask_dir, mask_type) for mask_type in self.assess_ranges
        ]
        self.to_img = transforms.ToPILImage()

    def load_model(self):
        if self.model_checkpoint_path != None:
            checkpoint = torch.load(self.model_checkpoint_path, map_location=self.device)
            if "model_state_dict" not in checkpoint:
                raise KeyError("Invalid checkpoint: Missing 'model_state_dict'")
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print("Model loaded successfully.")

    def create_output_dirs(self):
        for r in self.assess_ranges:
            outsub_folder_path = os.path.join(self.output_dir, r)
            os.makedirs(outsub_folder_path, exist_ok = True)
            self.out_dirs.append(outsub_folder_path)

    def to_image_save_format(self, processed_img):
        assert processed_img.min() >= -1 and processed_img.max() <= 1, "Not in range [-1, 1]"
        processed_img = (processed_img + 1) /  2 # => CHW: [0, 1]
        processed_img = processed_img.cpu() 
        out_img = self.to_img(processed_img) # PIL Image: HWC [0, 255]
        return out_img

    def save_image(self, processed_img, save_path):
        formatted_img = self.to_image_save_format(processed_img)
        formatted_img.save(save_path)
        print("Proccessed image saved at:", save_path)

    def gen(self):
        self.model.eval()
        for (mask_path, out_dir_path) in zip(self.mask_sub_folder_paths, self.out_dirs):
            test_dataset = InpaintingDataset(img_dir=self.img_dir, mask_dir=mask_path)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
            for index, (inputs, _) in enumerate(test_loader):
                imgs, masks = inputs[0].to(self.device), inputs[1].to(self.device)
                masks = transforms.Resize(size=(256, 256))(masks)
                outputs = self.model(imgs, masks)
                for i in range(outputs.shape[0]):
                    org_filename = test_dataset.imgs[index * self.batch_size + i]
                    save_path = os.path.join(out_dir_path, org_filename)
                    self.save_image(outputs[i], save_path)

