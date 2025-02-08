import torch
import numpy as np
import time
import os
from configs import *

def save_loss(file_name, epoch, model_loss, disc_loss):
    torch.save({
        'epoch': epoch,
        'model_loss': model_loss,
        'disc_loss': disc_loss
    }, file_name)

def save_checkpoint(file_name, epoch, model, optimizer):
    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, file_name)


def train(epochs, model, train_loader, criterion, optimizer, device, disc, disc_criterion, disc_optimizer, current_epoch=0):
    model.train()
    disc.train()
    lastest_checkpoint_time = time.time()
    print("Start training")
    for epoch in range(current_epoch, epochs):
        elapsed_time = time.time() - START_TIME
        if elapsed_time > DURATION:
            return  # Stop training

        model_total_loss = 0
        disc_total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            imgs, masks = inputs[0].to(device), inputs[1].to(device)
            targets = targets.to(device)
            outputs = model(imgs, masks)

            # Train Discriminator
            disc_optimizer.zero_grad()
            disc_fake_pred = disc(outputs.to(device))
            disc_real_pred = disc(targets.to(device))
            
            disc_fake_loss = disc_criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred).to(device))
            disc_real_loss = disc_criterion(disc_real_pred, torch.ones_like(disc_real_pred).to(device))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_total_loss += disc_loss.item()
            disc_loss.backward()
            disc_optimizer.step()

            # Train Model
            optimizer.zero_grad()
            outputs = model(imgs, masks)
            disc_fake_pred = disc(outputs.to(device))
            disc_loss = disc_criterion(disc_fake_pred, torch.ones_like(disc_fake_pred).to(device))
            loss = criterion(masks.to(device), outputs.to(device), targets.to(device), disc_loss)
            model_total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Save checkpoint
            if time.time() - lastest_checkpoint_time > SAVE_INTERVAL:
                lastest_checkpoint_time = time.time()
                model_checkpoint_dest = os.path.join(CHECKPOINTS_DIR, f'model_{epoch}.pth')
                disc_checkpoint_dest = os.path.join(CHECKPOINTS_DIR, f'disc_{epoch}.pth')

                save_checkpoint(model_checkpoint_dest, epoch, model, optimizer)
                save_checkpoint(disc_checkpoint_dest, epoch, disc, disc_optimizer)

        # Save loss
        loss_checkpoint_dest = os.path.join(CHECKPOINTS_DIR, f'loss_{epoch}.pth')
        save_loss(file_name= loss_checkpoint_dest,
                  epoch= epoch,
                  model_loss= model_total_loss / len(train_loader),
                  disc_loss= disc_total_loss / len(train_loader))


# ======================== Authors code starts here ========================
def _load(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location= torch.device('cpu'))
    return checkpoint

def load_checkpoint(path, model, optimizer=None, reset_optimizer=True, is_dis=False):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    if is_dis:
        s = checkpoint["disc"]
    else:
        s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s, strict=True)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    return model

def psnr(img1, img2):
    mse = np.mean((img1-img2)** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

