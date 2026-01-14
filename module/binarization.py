import torch

class SignBinarization(torch.autograd.Function):
    """
    Binarization function that converts input tensors to 0 and 1.
    """
    @staticmethod
    def forward(ctx, input):
        output = input.clone()
        output[input > 0] = 1
        output[input <= 0] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    
class SquarewaveBinarization(torch.autograd.Function):
    """
    Binarization function that converts input tensors to 0 and 1 with a cycle of 2.
    """
    @staticmethod
    def forward(ctx, input):
        output = input.clone()
        output[input % 2 < 1] = 1
        output[input % 2 >= 1] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input = torch.pi * torch.cos(2 * torch.pi * grad_input)
        return grad_input
    
class SignBinarization_sigmoid(torch.autograd.Function):
    """
    Binarization function that converts input tensors to 0 and 1.
    """
    @staticmethod
    def forward(ctx, input):
        output = input.clone()
        output[input > 0] = 1
        output[input <= 0] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input * (1 - grad_input)
        return grad_input
    
class kBitToFloat(torch.nn.Module):
    def __init__(self, k):
        super(kBitToFloat, self).__init__()
        self.k = k
        self.register_buffer('kernel', torch.tensor([2 ** i for i in range(self.k)], dtype=torch.float32) / (2 ** self.k - 1))

    def forward(self, x):
        """
        Convert k-bit binary representation to float.
        Args:
            x: Tensor of shape (..., n * k) with binary values (0 or 1)
        Returns:
            Tensor of shape (..., n) with float values
        """
        if x.dim() < 2 or x.size(-1) % self.k != 0:
            raise ValueError("Input tensor's last dimension must be divisible by k.")
        n = x.size(-1) // self.k
        x = x.view(*x.shape[:-1], n, self.k)
        x = x.float()
        x = x * self.kernel.to(x.device)
        x = x.sum(dim=-1)
        return x