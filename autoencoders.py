import torch
from torch import nn

"""
*******************************************************************************************************************************************************************************

AUTOENCODER CLASS

*******************************************************************************************************************************************************************************
"""

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, encoder_layer_dims, activation):
        super(Autoencoder, self).__init__()
        self.encoder = encoder(encoder_layer_dims, activation)
        self.decoder = decoder(list(reversed(encoder_layer_dims)), activation)

    def forward(self, x):
        return self.decoder(self.encoder(x))

"""
*******************************************************************************************************************************************************************************

ENCODERS

*******************************************************************************************************************************************************************************
"""

class MLPEncoder(nn.Module):
    def __init__(self, layer_dims, activation):
        super(MLPEncoder, self).__init__()

        self.layer_dims = layer_dims

        self.encoder = MLP(layer_dims, activation)

    def forward(self, x):
        return self.encoder(x)

"""
*******************************************************************************************************************************************************************************

DECODERS

*******************************************************************************************************************************************************************************
"""

class MLPDecoder(nn.Module):
    def __init__(self, layer_dims, activation):
        super(MLPDecoder, self).__init__()

        self.layer_dims = layer_dims

        self.decoder = MLP(layer_dims, activation)

    def forward(self, x):
        return self.decoder(x)

"""
*******************************************************************************************************************************************************************************

HELPERS

*******************************************************************************************************************************************************************************
"""

class MLP(nn.Module):
    def __init__(self, layer_dims, activation):
        super(MLP, self).__init__()

        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(activation)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)