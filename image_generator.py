import torch
import cv2
import argparse
from network.network_pro import Inpaint
from dataset.inpainting_dataset import InpaintingDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os

class ImageGen:
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
        if checkpoint != None:
            checkpoint = torch.load(self.model_checkpoint_path, map_location=self.device)
            if "model_state_dict" not in checkpoint:
                raise KeyError("Invalid checkpoint: Missing 'model_state_dict'")
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print("Model loaded successfully.")

    def create_output_dirs(self):
        for assess_range in self.assess_ranges:
            outsub_folder_path = os.path.join(self.output_dir, assess_range)
            os.makedirs(outsub_folder_path, exist_ok = True)
            self.out_dirs.append(outsub_folder_path)
          


    def save_image(self, processed_img, save_path):
        assert processed_img.min() >= -1 and processed_img.max() <= 1, "Not in range [-1, 1]"
        processed_img = torch.clamp(processed_img, -1, 1) # Make sure after tanh act
        processed_img = (processed_img + 1) /  2 # => [0, 1]
        processed_img = processed_img.permute(1, 2, 0).cpu()
        assert processed_img.min() >= 0 and processed_img.max() <= 1, "Not in range [0, 1]"
        final_img = self.to_img(processed_img)
        final_img.save(save_path)
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

# Example:
# (evaluation) python image_generator.py --ckpt <path_to_checkpoint> --img <path_to_image_folder> --mask <path_to_mask_folder> --output <path_to_output_folder> --batch_size 1   
# (test: not provide ckpt) python image_generator.py --img <path_to_image_folder> --mask <path_to_mask_folder> --output <path_to_output_folder> --batch_size 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inpaint inference")
    parser.add_argument("--ckpt", type=str, required=False, help="Path to model checkpoint")
    parser.add_argument("--img", type=str, required=True, help="Path to image folder")
    parser.add_argument("--mask", type=str, required=True, help="Path to mask folder")
    parser.add_argument("--output", type=str, required=True, help="Path to output folder")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    args = parser.parse_args()

    img_gen = ImageGen(
        model_checkpoint_path=args.ckpt,
        img_dir=args.img,
        mask_dir=args.mask,
        output_dir=args.output,
        batch_size=args.batch_size
    )
    with torch.no_grad():
        img_gen.load_model()

        img_gen.create_output_dirs()
        print("Output directories created successfully.")

        img_gen.gen()
        print("Image generation completed.")