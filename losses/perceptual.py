import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load the VGG16 model pre-trained on ImageNet
        self.weights = models.VGG16_Weights.IMAGENET1K_FEATURES
        self.vgg = models.vgg16(weights=self.weights).features.eval()
        
        # Default layers: Relu_1_2, Relu_2_2, Relu_3_3, Relu_4_3
        self.layers = [3, 8, 15, 22]
        
        # Freeze parameters of VGG16
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Get the transformation pipeline from the weights
        # self.transforms = models.VGG16_Weights.IMAGENET1K_FEATURES.transforms()

    def forward(self, input, target):
        """
        Compute perceptual loss.
        Args:
        - input (torch.Tensor): Generated image tensor (B, C, H, W).
        - target (torch.Tensor): Target image tensor (B, C, H, W).
        Returns:
        - loss (torch.Tensor): Perceptual loss value.
        """
        # Preprocess images using the predefined transforms
        # input = self.transforms(input)
        # target = self.transforms(target)
        
        input_features = input
        target_features = target
        loss = 0.0
        
        # Extract and compare features
        for i, layer in enumerate(self.vgg):
            input_features = layer(input_features)
            target_features = layer(target_features)
            if i in self.layers:
                loss += nn.functional.mse_loss(input_features, target_features)
        
        return loss