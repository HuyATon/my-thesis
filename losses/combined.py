import torch.nn as nn

from losses.perceptual import PerceptualLoss
from losses.reconstruction import ReconstructionLoss


class CombinedLoss(nn.Module):

    def __init__(self, w_mr = 1, w_p = 10, w_disc = 0.001):
        super(CombinedLoss, self).__init__()

        self.w_mr = w_mr
        self.w_p = w_p
        self.w_disc = w_disc

        self.perceptual = PerceptualLoss()
        self.reconstruction = ReconstructionLoss()
    
    def forward(self, mask, y_pred, y_true, disc_loss):

        p_loss = self.perceptual(y_pred, y_true)
        mr_loss = self.reconstruction(mask, y_pred, y_true)

        loss = self.w_mr * mr_loss + self.w_p * p_loss + self.w_disc * disc_loss
        return loss

