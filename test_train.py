import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from utils import *
from configs import *
from train_session import TrainSession

from network.network_pro import Inpaint
from losses.combined import CombinedLoss
from network.discriminator import Discriminator
from dataset.inpainting_dataset import InpaintingDataset

# Training configs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
dateset = InpaintingDataset(img_dir='samples/test_img', mask_dir='samples/test_mask')
train_loader = DataLoader(dateset, batch_size=1, shuffle=True, num_workers=0)

# Model and loss function
model = Inpaint().to(device)
criterion = CombinedLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Discriminator
disc = Discriminator().to(device)
disc_criterion = nn.BCEWithLogitsLoss().to(device)
disc_optimizer = torch.optim.Adam(disc.parameters(), lr=LR)

print("#model_params:", model.count_parameters())
# Training
model.train()
disc.train()

train_session = TrainSession(
        device=device,
        train_loader=train_loader,
        checkpoint_repo='.'
    )