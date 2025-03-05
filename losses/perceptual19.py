import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PerceptualLoss19(nn.Module):
    def __init__(self, normalize_inputs=False):
        super(PerceptualLoss19, self).__init__()
        
        self.normalize_inputs = normalize_inputs
        self.mean_ = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
        self.std_ = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]
        
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()

        
        for i, layer in enumerate(self.vgg):
            if isinstance(layer, nn.MaxPool2d):
                self.vgg[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        for param in self.vgg.parameters():
            param.requires_grad = False
        

        self.feature_layers = [i for i, layer in enumerate(self.vgg) if isinstance(layer, nn.ReLU)]

    def normalize(self, x):
        return (x - self.mean_.to(x.device)) / self.std_.to(x.device)

    def check_range(self, tensor, min_value, max_value, name):
        actual_min, actual_max = tensor.min(), tensor.max()
        if actual_min < min_value or actual_max > max_value:
            warnings.warn(f"{name} must be in {min_value}..{max_value} range, but it ranges {actual_min}..{actual_max}")

    def forward(self, input, target, mask=None):

        input = (input - input.min()) / (input.max() - input.min())
        target = (target - target.min()) / (target.max() - target.min())
        self.check_range(target, 0, 1, 'PerceptualLoss target')

        if self.normalize_inputs:
            input = self.normalize(input)
            target = self.normalize(target)
        
        loss = 0.0
        input_features, target_features = input, target
        
        for i, layer in enumerate(self.vgg):
            input_features = layer(input_features)
            target_features = layer(target_features)
            
            if i in self.feature_layers:
                l = F.mse_loss(input_features, target_features, reduction='none')
                
                if mask is not None:
                    cur_mask = F.interpolate(mask, size=input_features.shape[-2:], mode='bilinear', align_corners=False)
                    l = l * (1 - cur_mask)
                
                loss += l.mean()
        
        return loss