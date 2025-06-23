from train_session import TrainSession
import torch
import argparse
from dataset.inpainting_dataset import InpaintingDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="Training parameters")
parser.add_argument("--img_dir", type=str, required=True, help="Path to the image directory")
parser.add_argument("--mask_dir", type=str, required=True, help="Path to the mask directory")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
parser.add_argument("--model_checkpoint", type=str, default=None, help="Path to the model checkpoint")
parser.add_argument("--disc_checkpoint", type=str, default=None, help="Path to the discriminator checkpoint")
parser.add_argument("--repo_path", type=str, required=True, help="Path to save checkpoints")
parser.add_argument("--num_workers", type=int, default=0, help="#workers, remember to set CPUs")
args = parser.parse_args()

assert torch.cuda.is_available(), "CUDA is not available."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
train_dataset = InpaintingDataset(img_dir=args.img_dir, mask_dir=args.mask_dir)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers= args.num_workers) 

if __name__ == "__main__":
    train_session = TrainSession(
        device=device,
        train_loader=train_loader,
        checkpoint_repo=args.repo_path
    )
    train_session.prepare_repo()
    if args.model_checkpoint and args.disc_checkpoint:
        train_session.load_checkpoints(model_path = args.model_checkpoint, disc_path = args.disc_checkpoint)
        print("Checkpoints loaded successfully.")
    else:
        print("No checkpoints provided. Starting from scratch.")
    train_session.start_training()