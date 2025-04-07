import torch
import torch.nn as nn 
import os
import cv2
from torch.utils.data import Dataset, DataLoader
import time
from utils import *
from configs import *

from network.network_pro import Inpaint
from losses.combined import CombinedLoss
from network.discriminator import Discriminator
from dataset.inpainting_dataset import InpaintingDataset

# Training configs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

# Model and loss function
model = Inpaint().to(device)
criterion = CombinedLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Discriminator
disc = Discriminator().to(device)
disc_criterion = nn.BCEWithLogitsLoss().to(device)
disc_optimizer = torch.optim.Adam(disc.parameters(), lr=LR)

# Training
model.train()
disc.train()
for epoch in range(10):
        model_total_loss = 0
        disc_total_loss = 0
        inputs = (torch.randn(BATCH_SIZE, 3, 256, 256).to(device), torch.randn(BATCH_SIZE, 1, 256, 256).to(device))
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
