from train_session import TrainSession
import torch
import argparse
from dataset.inpainting_dataset import InpaintingDataset
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import os
from torch.distributed import init_process_group, destroy_process_group
from network.network_pro import Inpaint
from network.discriminator import Discriminator


parser = argparse.ArgumentParser(description="Training parameters")
parser.add_argument(
    "--img_dir", type=str, required=True, help="Path to the image directory"
)
parser.add_argument(
    "--mask_dir", type=str, required=True, help="Path to the mask directory"
)
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
parser.add_argument(
    "--model_checkpoint", type=str, default=None, help="Path to the model checkpoint"
)
parser.add_argument(
    "--disc_checkpoint",
    type=str,
    default=None,
    help="Path to the discriminator checkpoint",
)
parser.add_argument(
    "--repo_path", type=str, required=True, help="Path to save checkpoints"
)
parser.add_argument(
    "--num_workers", type=int, default=0, help="#workers, remember to set CPUs"
)
args = parser.parse_args()

assert torch.cuda.is_available(), "CUDA is not available."

def ddp_setup(rank: int, world_size: int):
    """
    args:
        rank: int, rank of the current process
        world_size: int, total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def load_train_objs():
    dataset = InpaintingDataset(img_dir=args.img_dir, mask_dir=args.mask_dir)

    model = Inpaint()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    disc = Discriminator()
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=1e-4)

    return dataset, model, model_optimizer, disc, disc_optimizer


def prepare_dataloader(dataset: InpaintingDataset, batch_size: int):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset=dataset, shuffle=False),
    )


def main(rank: int, world_size: int):
    ddp_setup(rank, world_size)
    dataset, model, model_optimizer, disc, disc_optimizer = load_train_objs()
    train_loader = prepare_dataloader(dataset=dataset, batch_size=int(args.batch_size))
    train_session = TrainSession(
        rank=rank,
        model=model,
        model_optimizer=model_optimizer,
        disc=disc,
        disc_optimizer=disc_optimizer,
        train_loader=train_loader,
        checkpoint_repo=args.repo_path,
    )

    if args.model_checkpoint and args.disc_checkpoint:
        train_session.load_checkpoints(
            model_path=args.model_checkpoint, disc_path=args.disc_checkpoint
        )
        print(f"[GPU {rank}] - Checkpoints loaded successfully.")
    else:
        print(f"[GPU {rank}] - No checkpoints provided, starting from scratch.")

    train_session.prepare_repo()
    train_session.start_training()
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
