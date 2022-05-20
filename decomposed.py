"""decomposed.py - decomposed model and module class for EfficientNet.
   Built to create new encoder-decoder architecture
"""

# Author: Enrique Maravilla


import torch
from torch import nn
from torch.nn.functional import interpolate

class Autoencoder(nn.Module):
    """EfficientNet model decomposition to encoder decoder architecture

    Args:

    """
    def __int__(self):
        super().__init__()

    def create_decoder(self, encoder_list, input_size) -> torch.Module:
        input_tensor = torch.randn(1, 3, 224, 224)

        decoder = nn.Sequential(
            # interpolate(input_tensor, (32, 112, 112), mode='bilinear')
            nn.Upsample(size=(3, 225, 225), mode='bilinear'),
            nn.Upsample(size=(16, 112, 112), mode='bilinear'),
            nn.Upsample(size=(32, 112, 112), mode='bilinear'),
            nn.Upsample(size=(96, 56, 56), mode='bilinear'),
            nn.Upsample(size=(144, 56, 56), mode='bilinear'),
            nn.Upsample(size=(240, 28, 28), mode='bilinear'),
            nn.Upsample(size=(480, 14, 14), mode='bilinear'),
            nn.Upsample(size=(672, 14, 14), mode='bilinear'),
            nn.Upsample(size=(1152, 7, 7), mode='bilinear')
        )
        return decoder

    def create_encoder(self, model: torch.Module) -> torch.Module:
        # eliminate or replace the classification layers in model
        eliminated_layers = torch.nn.Sequential(
            model._conv_head,
            model._bn1,
            model._avg_pooling,
            model._dropout,
            model._fc,
            model._swish
        )
        encoder = torch.nn.Sequential(
            model._conv_stem,
            model._bn0,
            model._blocks
        )
        return encoder, eliminated_layers

    def decompose_encoder(self, encoder):
        encoder_list = []
        encoder0 = torch.nn.Identity()
        encoder_list.append(encoder0)

        encoder1 = torch.nn.Sequential(
            next(encoder.children),
            next(encoder.children)
        )

        encoder_list.append(encoder1)

        block_iterator = encoder['3'].children #._blocks

        for block in block_iterator:
            encoder_list.append(block)

        return encoder_list

    @staticmethod
    def put_model_back_together(self, complete_encoder, eliminated_layers):
        return torch.nn.Sequential(complete_encoder, eliminated_layers)

