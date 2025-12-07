import torch

class SAE(torch.nn.Module):
    def __init__(
        self, 
        input_dim,
        hidden_dim,
        output_dim = None,
        bias = True,
        inner_activation = 'relu',
        output_activation = 'relu',
        topk = 10,  # For 'topk' activation
        theta = 0.5,  # For 'jumprelu' activation
    ):
        super(SAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.output_activation = output_activation
        self.topk = topk
        self.theta = theta

        # Encoder
        self.encoder = torch.nn.Linear(self.input_dim, self.hidden_dim)
        # Decoder
        self.decoder = torch.nn.Linear(self.hidden_dim, self.output_dim, bias=bias)

        if inner_activation == 'sigmoid':
            self.inner_activation = torch.nn.Sigmoid()
        elif inner_activation == 'tanh':
            self.inner_activation = torch.nn.Tanh()
        elif inner_activation == 'relu':
            self.inner_activation = torch.nn.ReLU()
        elif inner_activation == 'none':
            self.inner_activation = lambda x: x
        elif inner_activation == 'jumprelu':
            self.inner_activation = lambda x: torch.where(x > theta, x, torch.zeros_like(x))
        elif inner_activation == 'topk':
            self.inner_activation = lambda x: x * (x >= torch.topk(x, topk, dim=1).values[:, -1].unsqueeze(1)).float()
        else:
            raise ValueError(f"Unsupported inner activation: {inner_activation}")

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
        x_b = self.inner_activation(x)

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