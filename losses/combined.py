import torch.nn as nn

from losses.perceptual import PerceptualLoss
from losses.reconstruction import ReconstructionLoss
from losses.perceptual19 import PerceptualLoss19
from losses.adversarial import NonSaturatingWithR1 


class CombinedLoss(nn.Module):

    def __init__(self, w_mr=1, w_p=10, w_disc=0.001, w_adv=0.1):
        super(CombinedLoss, self).__init__()

        self.w_mr = w_mr
        self.w_p = w_p
        self.w_disc = w_disc
        self.w_adv = w_adv  
        # self.perceptual = PerceptualLoss()
        self.perceptual = PerceptualLoss19()
        self.reconstruction = ReconstructionLoss()
        self.adversarial_loss = NonSaturatingWithR1()  # Adversarial loss

    def forward(self, mask, y_pred, y_true, disc_pred_fake, disc_loss):


        p_loss = self.perceptual(y_pred, y_true)
        mr_loss = self.reconstruction(mask, y_pred, y_true)
        adv_loss = self.adversarial_loss.generator_loss(disc_pred_fake, mask)

        loss = (
            self.w_mr * mr_loss +
            self.w_p * p_loss +
            self.w_disc * disc_loss +
            self.w_adv * adv_loss
        )
        return loss