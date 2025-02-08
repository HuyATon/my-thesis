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


# Data loader
train_dataset = InpaintingDataset(img_dir=IMG_DIR, mask_dir=MASK_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

# Model and loss function
model = Inpaint().to(device)
criterion = CombinedLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Discriminator
disc = Discriminator().to(device)
disc_criterion = nn.BCEWithLogitsLoss().to(device)
disc_optimizer = torch.optim.Adam(disc.parameters(), lr=LR)

# Load checkpoint
current_epoch = 0
if MODEL_CHECKPOINT != None and DISC_CHECKPOINT != None:
    model_checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
    model.load_state_dict(model_checkpoint["model_state_dict"])
    optimizer.load_state_dict(model_checkpoint["optimizer_state_dict"])

    disc_checkpoint = torch.load(DISC_CHECKPOINT, map_location=device)
    disc.load_state_dict(disc_checkpoint["model_state_dict"])
    disc_optimizer.load_state_dict(disc_checkpoint["optimizer_state_dict"])

    current_epoch = model_checkpoint["epoch"] + 1
    print("Loaded")

# Training
train(EPOCHS, model, train_loader, criterion, optimizer, device, disc, disc_criterion, disc_optimizer, current_epoch)
