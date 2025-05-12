import torch.nn as nn
import torch
import torchvision.models as models

# class PerceptualLoss(nn.Module):
#     def __init__(self):
#         super(PerceptualLoss, self).__init__()
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.means = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
#         self.stds = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
#         features = models.vgg16(pretrained=True).features
#         self.to_relu_1_2 = nn.Sequential()
#         self.to_relu_2_2 = nn.Sequential()
#         self.to_relu_3_3 = nn.Sequential()
#         self.to_relu_4_3 = nn.Sequential()
        
#         for x in range(4):
#             self.to_relu_1_2.add_module(str(x), features[x])
#         for x in range(4, 9):
#             self.to_relu_2_2.add_module(str(x), features[x])
#         for x in range(9, 16):
#             self.to_relu_3_3.add_module(str(x), features[x])
#         for x in range(16, 23):
#             self.to_relu_4_3.add_module(str(x), features[x])

#         for param in self.parameters():
#             param.requires_grad = False

#         self.standard_transform = models.VGG16_Weights.DEFAULT.transforms()

#     def calculate_features(self, x):
#         out = []
#         h = self.to_relu_1_2(x)
#         out.append(h)
#         h = self.to_relu_2_2(h)
#         out.append(h)
#         h = self.to_relu_3_3(h)
#         out.append(h)
#         h = self.to_relu_4_3(h)
#         out.append(h)
#         return out

#     def normalize(self, x, from_tanh=True):
#         if from_tanh:
#             x = (x + 1) / 2 # [-1, 1] => [0, 1]
#         return self.standard_transform(x)

#     def forward(self, input, target):
#         """
#         Compute perceptual loss.
#         Args:
#         - input (torch.Tensor): Generated image tensor (B, C, H, W).
#         - target (torch.Tensor): Target image tensor (B, C, H, W).
#         Returns:
#         - loss (torch.Tensor): Perceptual loss value.
#         """
#         normalized_input = self.normalize(input)
#         normalized_target = self.normalize(target)
#         loss = 0.0
#         input_out = self.calculate_features(normalized_input)
#         target_out = self.calculate_features(normalized_target)
#         for i in range(len(input_out)):
#             loss += nn.functional.mse_loss(input_out[i], target_out[i])
#         return loss
    

class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights
        self.transform = models.VGG19_Weights.IMAGENET1K_V1.transforms()

    def normalize(self, x, from_tanh=True):
        if from_tanh:
            x = (x + 1) / 2 # [-1, 1] => [0, 1]
        return self.standard_transform(x)

    def __call__(self, x, y):
        # Compute features
        x = self.transform(self.normalize(x))
        y = self.transform(self.normalize(y))
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss



class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out
