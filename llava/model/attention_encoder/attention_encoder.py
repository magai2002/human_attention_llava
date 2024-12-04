import torch
import torch.nn.functional as F
from torchvision import models
from torch import nn

class ModifiedResNetVisualEncoder(nn.Module):
    def __init__(self, base_model):
        super(ModifiedResNetVisualEncoder, self).__init__()
        # Use all layers of ResNet18 except the final fully connected layer
        self.base = nn.Sequential(*list(base_model.children())[:-2])  # Retain until the second-to-last layer
        
        # Add a 1x1 convolution layer to project to a single channel (for saliency map)
        self.conv1x1 = nn.Conv2d(512, 1, kernel_size=1)  # 512 is the output channel size of ResNet18’s last conv layer
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()  # To normalize output to [0, 1]
        
    def forward(self, x):
        # Pass input through the modified base model
        x = self.base(x)  # Output should have shape (batch_size, 512, H, W)
        
        x = self.conv1x1(x)  # Reduce channel dimensions to 1
        x = self.upsample(x)  # Upsample to match the input dimensions (224x224)
        x = self.sigmoid(x)  # Output normalized to [0, 1]
        return x.squeeze(0)

class AttentionModule(nn.Module):
    def __init__(self, weights_path, device='cuda'):
        super(AttentionModule, self).__init__()
        
        # Load base ResNet18 model and initialize ModifiedResNetVisualEncoder
        base_model = models.resnet18(pretrained=True)
        self.attention_encoder = ModifiedResNetVisualEncoder(base_model)
        
        # Load the specified weights into the attention encoder
        self.attention_encoder.load_state_dict(torch.load(weights_path, map_location=device))
        
        # Move attention encoder to the specified device and set to evaluation mode
        self.attention_encoder = self.attention_encoder.to(device).eval()
        self.device = device

    def forward(self, image_features):
        print("Attention module applied.")
        # Generate the attention map using the modified ResNet-based human attention encoder
        with torch.no_grad():
            # Resize image_features to match the input dimensions of ResNet18 if needed
            input_for_attention = F.interpolate(image_features, size=(224, 224), mode='bilinear', align_corners=False)
            
            # Pass through the attention encoder to get the saliency map
            attention_features = self.attention_encoder(input_for_attention)  # Output shape: (batch_size, 1, 224, 224)

        return attention_features

