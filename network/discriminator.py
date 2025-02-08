import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2))
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1))
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2))
        self.conv5 = nn.utils.spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1))
        self.conv6 = nn.utils.spectral_norm(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2))
        self.conv7 = nn.utils.spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1))
        self.linear = nn.utils.spectral_norm(nn.Linear(in_features = 346112, out_features= 1))
        self.slope = 0.1

        self.disc = nn.Sequential(
            self.conv1,
            nn.LeakyReLU(self.slope),
            self.conv2,
            nn.LeakyReLU(self.slope),
            self.conv3,
            nn.LeakyReLU(self.slope),
            self.conv4,
            nn.LeakyReLU(self.slope),
            self.conv5,
            nn.LeakyReLU(self.slope),
            self.conv6,
            nn.LeakyReLU(self.slope),
            self.conv7,
            nn.LeakyReLU(self.slope),
            nn.Flatten(),
            self.linear
        )

    def forward(self, x):
        x = self.disc(x)
        return x



