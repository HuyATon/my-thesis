from train_session import TrainSession
import torch
import argparse
from dataset.inpainting_dataset import InpaintingDataset
from torch.utils.data import DataLoader


# Create argument parser for: 
# --img_dir
# --mask_dir
# --batch_size
# --model_checkpoint
# --disc_checkpoint
# --repo_path

# MANUAL: python --img_dir <path> --mask_dir <path> --batch_size <int> --model_checkpoint <path> --disc_checkpoint <path> --repo_path <path>
parser = argparse.ArgumentParser(description="Training parameters")
parser.add_argument("--img_dir", type=str, required=True, help="Path to the image directory")
parser.add_argument("--mask_dir", type=str, required=True, help="Path to the mask directory")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
parser.add_argument("--model_checkpoint", type=str, default=None, help="Path to the model checkpoint")
parser.add_argument("--disc_checkpoint", type=str, default=None, help="Path to the discriminator checkpoint")
parser.add_argument("--repo_path", type=str, required=True, help="Path to save checkpoints")

args = parser.parse_args()
# Assign arguments to variables
img_dir = args.img_dir
mask_dir = args.mask_dir
batch_size = args.batch_size
model_checkpoint = args.model_checkpoint
disc_checkpoint = args.disc_checkpoint
repo_path = args.repo_path

assert torch.cuda.is_available(), "CUDA is not available."
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
train_dataset = InpaintingDataset(img_dir=img_dir, mask_dir=mask_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size) 

if __name__ == "__main__":
    train_session = TrainSession(
        device=device,
        train_loader=train_loader,
        checkpoint_repo=repo_path
    )
    train_session.prepare_repo()
    if model_checkpoint and disc_checkpoint:
        train_session.load_checkpoints(model_checkpoint, disc_checkpoint)
        print("Checkpoints loaded successfully.")
    else:
        print("No checkpoints provided. Starting from scratch.")
    train_session.start_training()