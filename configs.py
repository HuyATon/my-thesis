import time

# Train configs
EPOCHS = 9999999
BATCH_SIZE = 8
LR = 1e-3
IMG_DIR = '/media02/nnthao05/data/celeba_hq_256'
MASK_DIR = '/media02/nnthao05/data/celeba_hq_256_mask'
CHECKPOINTS_DIR = '/media02/nnthao05/code/cmt_git/checkpoints/v3' # Change this to new folder
MODEL_CHECKPOINT = '/media02/nnthao05/code/cmt_git/checkpoints/v2/model_50.pth' # Change this to current checkpoint (None if not exist)
DISC_CHECKPOINT = '/media02/nnthao05/code/cmt_git/checkpoints/v2/disc_50.pth' # Change this to current checkpoint (None if not exist)

# Time configs
DURATION = 48 * 60 * 60  # ~ 2 days
SAVE_INTERVAL = 60 * 60
START_TIME = time.time()
ONE_DAY = 24 * 60 * 60