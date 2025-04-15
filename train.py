from utils_classes.train_session import TrainSession
import torch
from configs import *
from dataset.inpainting_dataset import InpaintingDataset
from torch.utils.data import DataLoader


assert torch.cuda.is_available(), "CUDA is not available."
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
train_dataset = InpaintingDataset(img_dir=IMG_DIR, mask_dir=MASK_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE) 

train_session = TrainSession(device = device,
                             model_checkpoint=MODEL_CHECKPOINT,
                             disc_checkpoint=DISC_CHECKPOINT,
                             train_loader=train_loader,
                             checkpoint_repo=CHECKPOINTS_DIR,
                             lr=LR
                             )
train_session.load_checkpoints()
train_session.start_training()