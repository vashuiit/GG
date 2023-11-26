import torch
import torch.nn as nn
import torchvision.models as models

class StyleTransferModel(nn.Module):
    def __init__(self, content_layers, style_layers):
        super(StyleTransferModel, self).__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers

        # Load a pre-trained model (VGG16 in this case)
        self.vgg = models.vgg16(pretrained=True).features.eval()

    def forward(self, x):
        content_outputs = []
        style_outputs = []

        for layer_idx, layer in enumerate(self.vgg):
            x = layer(x)

            if layer_idx in self.content_layers:
                content_outputs.append(x)

            if layer_idx in self.style_layers:
                style_outputs.append(x)

        return content_outputs, style_outputs
