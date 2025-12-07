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