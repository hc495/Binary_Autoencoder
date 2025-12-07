from . import binarization
import torch

class BinarizedAutoencoder(torch.nn.Module):
    def __init__(
        self, 
        input_dim,
        hidden_dim,
        output_dim = None,
        bias = True,
        binarization_type = 'sign',
        output_activation = 'relu',
        cycle_for_squarewave = 2,
    ):
        super(BinarizedAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.binarization_type = binarization_type
        self.output_activation = output_activation
        self.cycle_for_squarewave = cycle_for_squarewave

        # Encoder
        self.encoder = torch.nn.Linear(self.input_dim, self.hidden_dim)
        
        if binarization_type == 'sign':
            self.binarization = binarization.SignBinarization
        elif binarization_type == 'squarewave':
            self.binarization = binarization.SquarewaveBinarization
        elif binarization_type == 'sign_s':
            self.binarization = binarization.SignBinarization_sigmoid

        # Decoder
        self.decoder = torch.nn.Linear(self.hidden_dim, self.output_dim, bias=bias)

        if output_activation == 'sigmoid':
            self.output_activation = torch.nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_activation = torch.nn.Tanh()
        elif output_activation == 'relu':
            self.output_activation = torch.nn.ReLU()
        elif output_activation == 'none':
            self.output_activation = lambda x: x
        else:
            raise ValueError(f"Unsupported output activation: {output_activation}")
    
    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        x_b = self.binarization.apply(x)

        # Decoder
        y = self.decoder(x_b)
        y = self.output_activation(y)

        return y, x_b
    
    def encode(self, x):
        y, x_b = self.forward(x)
        return x_b
    
    def decode(self, x_b):
        # Decode the binarized representation
        y = self.decoder(x_b)
        y = self.output_activation(y)
        return y