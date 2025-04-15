import torch.nn as nn
import torch
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.means = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.stds = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        
        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.standard_transform = models.VGG16_Weights.DEFAULT.transforms()

    def calculate_features(self, x):
        out = []
        h = self.to_relu_1_2(x)
        out.append(h)
        h = self.to_relu_2_2(h)
        out.append(h)
        h = self.to_relu_3_3(h)
        out.append(h)
        h = self.to_relu_4_3(h)
        out.append(h)
        return out

    def normalize(self, x, from_tanh=True):
        if from_tanh:
            x = (x + 1) / 2 # [-1, 1] => [0, 1]
        return self.standard_transform(x)

    def forward(self, input, target):
        """
        Compute perceptual loss.
        Args:
        - input (torch.Tensor): Generated image tensor (B, C, H, W).
        - target (torch.Tensor): Target image tensor (B, C, H, W).
        Returns:
        - loss (torch.Tensor): Perceptual loss value.
        """
        normalized_input = self.normalize(input)
        normalized_target = self.normalize(target)
        loss = 0.0
        input_out = self.calculate_features(normalized_input)
        target_out = self.calculate_features(normalized_target)
        for i in range(len(input_out)):
            loss += nn.functional.mse_loss(input_out[i], target_out[i])
        return loss