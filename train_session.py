import torch
import os
import time
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from losses.combined import CombinedLoss
from network.network_pro import Inpaint
from network.discriminator import Discriminator
from utils import save_checkpoint, save_loss
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


MODEL_LOSS_KEY = "model_loss"
DESC_LOSS_KEY = "disc_loss"
EPOCH_KEY = "epoch"
WEIGHT_KEY = "model_state_dict"
OPT_KEY = "optimizer_state_dict"


class TrainSession:
    def __init__(
        self,
        rank: int,
        model: Inpaint,
        model_optimizer: torch.optim.Optimizer,
        disc: Discriminator,
        disc_optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        checkpoint_repo: str,
        duration=48 * 60 * 60,  # 2 days
        save_window=60 * 60,  # 1 hour
    ):
        self.start_time = time.time()
        self.latest_checkpoint_time = time.time()
        self.max_epoch = 9999999
        self.last_epoch = 0
        self.rank = rank
        self.train_loader = train_loader
        self.checkpoint_repo = checkpoint_repo
        self.duration = duration
        self.save_window = save_window

        self.model = model.to(rank)
        self.model = DDP(self.model, device_ids=[rank])

        self.optimizer = model_optimizer
        self.criterion = CombinedLoss().to(rank)

        self.disc = disc.to(rank)
        self.disc = DDP(self.disc, device_ids=[rank])
        self.disc_optimizer = disc_optimizer

        self.mask_transform = transforms.Compose(
            [
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
            ]
        )

    def prepare_repo(self):
        for repo in ["loss", "model", "disc", "log_img"]:
            repo_path = os.path.join(self.checkpoint_repo, repo)
            if not os.path.exists(repo_path):
                os.makedirs(repo_path)

    def load_checkpoints(self, model_path: str, disc_path: str):
        saved_checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(saved_checkpoint[WEIGHT_KEY])
        self.optimizer.load_state_dict(saved_checkpoint[OPT_KEY])
        saved_checkpoint = torch.load(disc_path, map_location=self.device)
        self.disc.load_state_dict(saved_checkpoint[WEIGHT_KEY])
        self.disc_optimizer.load_state_dict(saved_checkpoint[OPT_KEY])

        self.last_epoch = saved_checkpoint[EPOCH_KEY] + 1

    def start_training(self):
        self.model.train()
        self.disc.train()
        for epoch in range(self.last_epoch, self.max_epoch):
            progress = time.time() - self.start_time
            if progress > self.duration:
                return  # stop training
            model_total_loss = 0
            disc_total_loss = 0
            for inputs, gt in self.train_loader:
                imgs, masks = inputs[0].to(self.device), inputs[1].to(self.device)
                gt = gt.to(self.device)
                masks = self.mask_transform(masks)

                fakes = self.model(imgs, masks)

                # Train Discriminator
                self.disc_optimizer.zero_grad()
                fake_pred = self.disc(fakes.detach())
                real_pred = self.disc(gt)
                real_loss = torch.mean(torch.nn.ReLU()(1.0 - real_pred))
                fake_loss = torch.mean(torch.nn.ReLU()(1.0 + fake_pred))
                disc_loss = fake_loss + real_loss
                disc_total_loss += disc_loss.item()
                disc_loss.backward()
                self.disc_optimizer.step()

                # Train Model
                self.optimizer.zero_grad()
                fake_pred = self.disc(fakes)
                disc_loss = -torch.mean(torch.nn.ReLU()(fake_pred))
                loss = self.criterion(masks, fakes, gt, disc_loss)
                model_total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            if epoch % 10 == 0 and self.rank == 0:
                self.log_outputs(epoch, fakes=fakes, gt=gt, masks=masks)

            if self.rank == 0:
                self.perform_logging(
                    epoch,
                    model_total_loss / len(self.train_loader),
                    disc_total_loss / len(self.train_loader),
                )

    def perform_logging(self, epoch: int, model_loss: float, disc_loss: float):
        self.log_loss(epoch, model_loss, disc_loss)
        if (
            time.time() - self.latest_checkpoint_time > self.save_window
            and time.time() - self.start_time > 24 * 60 * 60
        ):
            self.latest_checkpoint_time = time.time()
            self.log_checkpoint(epoch=epoch)

    def log_checkpoint(self, epoch: int):
        model_checkpoint_dest = os.path.join(
            self.checkpoint_repo, "model", f"model_{epoch}.pth"
        )
        disc_checkpoint_dest = os.path.join(
            self.checkpoint_repo, "disc", f"disc_{epoch}.pth"
        )
        save_checkpoint(model_checkpoint_dest, epoch, self.model.module, self.optimizer)
        save_checkpoint(
            disc_checkpoint_dest, epoch, self.disc.module, self.disc_optimizer
        )

    def log_loss(self, epoch: int, model_loss: float, disc_loss: float):
        print(
            "[EPOCH {}]: gen_loss: {:.4f}, disc_loss : {:.4f}".format(
                str(epoch).zfill(5), model_loss, disc_loss
            )
        )
        loss_checkpoint_dest = os.path.join(
            self.checkpoint_repo, "loss", f"loss_{str(epoch).zfill(5)}.pth"
        )
        save_loss(loss_checkpoint_dest, epoch, model_loss, disc_loss)

    def log_outputs(
        self, epoch: int, fakes: torch.Tensor, gt: torch.Tensor, masks: torch.Tensor
    ):
        fakes = (fakes + 1) / 2
        masked = (gt + 1) / 2 * (1 - masks)
        log_img = make_grid(
            torch.cat([masked, fakes], dim=0),
            nrow=min(fakes.shape[0], self.train_loader.batch_size),
        )
        log_img = transforms.ToPILImage()(log_img.detach().cpu())
        img_path = os.path.join(
            self.checkpoint_repo, "log_img", f"log_{str(epoch).zfill(5)}.png"
        )
        log_img.save(img_path)
