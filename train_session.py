import torch
import os
import time
from network.network_pro import Inpaint
from network.discriminator import Discriminator
from utils import save_checkpoint, save_loss
from torch.utils.data import DataLoader

MODEL_LOSS_KEY = "model_loss"
DESC_LOSS_KEY = "disc_loss"
EPOCH_KEY = "epoch"
WEIGHT_KEY = "model_state_dict"
OPT_KEY = "optimizer_state_dict"

class TrainSession:
    def __init__(self, 
                 device: str, 
                 model_checkpoint: str, 
                 disc_checkpoint: str, 
                 train_loader: DataLoader,
                 checkpoint_repo: str,
                 lr= 1e-3, 
                 duration= 48 * 60 * 60, # 2 days
                 save_window= 60 * 60 # 1 hour
                 ):
        self.max_epoch = 9999999
        self.current_epoch = 0

        self.device = device
        self.model_checkpoint = model_checkpoint
        self.disc_checkpoint = disc_checkpoint
        self.checkpoint_repo = checkpoint_repo
        self.train_loader = train_loader
        self.lr = lr
        self.duration = duration
        self.save_window = save_window
      
        self.model = Inpaint().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.disc = Discriminator().to(self.device)
        self.disc_optimizer = torch.optim.Adam(self.disc.parameters(), lr=self.lr)

    
    def load_checkpoints(self):
        if self.model_checkpoint != None and self.disc_checkpoint != None:
            saved_checkpoint = torch.load(self.model_checkpoint, map_location=self.device)
            self.model.load_state_dict(saved_checkpoint[WEIGHT_KEY])
            self.optimizer.load_state_dict(saved_checkpoint[OPT_KEY])
            saved_checkpoint = torch.load(self.disc_checkpoint, map_location=self.device)
            self.disc.load_state_dict(saved_checkpoint[WEIGHT_KEY])
            self.disc_optimizer.load_state_dict(saved_checkpoint[OPT_KEY])
            self.current_epoch = saved_checkpoint[EPOCH_KEY]  +  1
            saved_checkpoint = None # free memory

    def save_checkpoint(self, epoch):
        model_checkpoint_dest = os.path.join(self.checkpoint_repo, f'model_{epoch}.pth')
        disc_checkpoint_dest = os.path.join(self.checkpoint_repo, f'disc_{epoch}.pth')
        save_checkpoint(model_checkpoint_dest, epoch, self.model, self.optimizer)
        save_checkpoint(disc_checkpoint_dest, epoch, self.disc, self.disc_optimizer)
    
    def save_loss(self, epoch, model_loss, disc_loss):
        loss_checkpoint_dest = os.path.join(self.checkpoint_repo, f'loss_{epoch}.pth')
        save_loss(loss_checkpoint_dest, epoch, model_loss, disc_loss)


    def start_training(self):
        start_time = time.time()
        lastest_checkpoint_time = time.time()
        self.model.train()
        self.disc.train()
        for epoch in range(self.current_epoch, self.max_epoch):
            progress = time.time() - start_time
            if progress > self.duration:
                return
            model_total_loss = 0
            disc_total_loss = 0
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                imgs, masks = inputs[0].to(self.device), inputs[1].to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(imgs, masks)

                # Train Discriminator
                self.disc_optimizer.zero_grad()
                disc_fake_pred = self.disc(outputs.to(self.device))
                disc_real_pred = self.disc(targets.to(self.device))
                disc_fake_loss = self.disc_criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred).to(self.device))
                disc_real_loss = self.disc_criterion(disc_real_pred, torch.ones_like(disc_real_pred).to(self.device))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                disc_total_loss += disc_loss.item()
                disc_loss.backward()
                self.disc_optimizer.step()

                # Train Model
                self.optimizer.zero_grad()
                outputs = self.model(imgs, masks)
                disc_fake_pred = self.disc(outputs.to(self.device))
                disc_loss = self.disc_criterion(disc_fake_pred, torch.ones_like(disc_fake_pred).to(self.device))
                loss = self.criterion(masks.to(self.device), outputs.to(self.device), targets.to(self.device), disc_loss)
                model_total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            self.save_loss(epoch = epoch, 
                        model_loss = model_total_loss / len(self.train_loader),
                        disc_loss = disc_total_loss / len(self.train_loader))
            if time.time() - lastest_checkpoint_time > self.save_window:
                self.save_checkpoint(epoch=epoch)
                lastest_checkpoint_time = time.time()
            self.current_epoch += 1

  