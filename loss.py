import torch
import torch.nn as nn
import torchvision.models as models

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps)
        return torch.mean(loss)

class PerceptualLoss(nn.Module):
    """VGG19 features based perceptual loss"""
    def __init__(self, layer_weights=None, use_input_norm=True):
        super(PerceptualLoss, self).__init__()
        self.use_input_norm = use_input_norm
        
        # Load VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Extract features until specific layers
        # relu5_4 is often used for texture
        # relu2_2, relu3_4, relu4_4, relu5_4
        
        # Let's use a simpler setup: VGG19 features up to relu5_4
        # We will use the 'features' sequential module
        
        # Indices for VGG19 (batch_norm=False):
        # relu1_2: 3
        # relu2_2: 8
        # relu3_4: 17
        # relu4_4: 26
        # relu5_4: 35
        
        self.features = vgg.features
        self.layer_indices = {'relu1_2': 3, 'relu2_2': 8, 'relu3_4': 17, 'relu4_4': 26, 'relu5_4': 35}
        
        if layer_weights is None:
            self.layer_weights = {'relu5_4': 1.0}
        else:
            self.layer_weights = layer_weights
            
        for param in self.parameters():
            param.requires_grad = False
            
        # Normalization for ImageNet
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        # x, y are in [0, 1] range
        if self.use_input_norm:
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std
            
        loss = 0
        x_in = x
        y_in = y
        
        max_idx = max([self.layer_indices[k] for k in self.layer_weights.keys()])
        
        for i, (name, module) in enumerate(self.features._modules.items()):
            x_in = module(x_in)
            y_in = module(y_in)
            
            # Check if this layer matches any of our targets
            for key, idx in self.layer_indices.items():
                if int(i) == idx and key in self.layer_weights:
                    loss += self.layer_weights[key] * torch.mean(torch.abs(x_in - y_in))
            
            if int(i) >= max_idx:
                break
                
        return loss

class CompositeLoss(nn.Module):
    def __init__(self, charb_weight=1.0, percept_weight=0.01):
        super(CompositeLoss, self).__init__()
        self.charb = CharbonnierLoss()
        self.percept = PerceptualLoss()
        self.charb_weight = charb_weight
        self.percept_weight = percept_weight
        
    def forward(self, sr, hr):
        l_char = self.charb(sr, hr)
        l_per = self.percept(sr, hr)
        return self.charb_weight * l_char + self.percept_weight * l_per, l_char, l_per
