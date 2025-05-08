"""
Author: Elias Mapendo
Date: April 20, 2025
Description:
Defines the Gradient Reversal Layer (GRL) used for adversarial domain adaptation.
The GRL acts as an identity during the forward pass and multiplies the gradient by -λ during backpropagation.
"""

from torch.autograd import Function
import torch.nn as nn

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        """
        Args:
            ctx: Context object to save information for backward computation.
            x (Tensor): Input tensor.
            lambda_ (float): The scaling factor for the gradient reversal.

        Returns:
            Tensor: Same as input.
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            ctx: Context object containing saved lambda_.
            grad_output (Tensor): Gradient flowing from upstream.

        Returns:
            Tuple: Reversed gradient tensor and None (no gradient for lambda_).
        """
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    """
    Args:
        x (Tensor): Input tensor.
        lambda_ (float): Gradient reversal strength.

    Returns:
        Tensor: Output after applying gradient reversal.
    """
    return GradReverse.apply(x, lambda_)


class GradientReversalLayer(nn.Module):
    """
    A PyTorch nn.Module wrapper around the GradReverse autograd function.
    This can be inserted into any model as a normal layer.
    """
    def __init__(self, lambda_=1.0):
        """
        Initializes the Gradient Reversal Layer.

        Args:
            lambda_ (float): How strongly to reverse the gradients.
        """
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x, lambda_=None):
        """
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after gradient reversal during backprop.
            :param x:
            :param lambda_:
        """
        return grad_reverse(x, lambda_ if lambda_ is not None else self.lambda_)
