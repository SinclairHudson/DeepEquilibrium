import torch
import torch.nn as nn
from rootfind import BroydenRootFind

class DEQ(nn.Module):
    """
    A module to represent a DEQ.
    """
    def __init__(self, f, eq_size, forward_eps, backward_eps, alpha, max_iters):
        """
        :param forward_eps: epsilon for the equilibrium in the forward pass
        :param backward_eps: epsilon for the equilibrium in the backward pass
        :param f: The Pytorch differentiable function that will be "iterated".
        This could be a nn.Module, or a Functional layer. Should have parameters
        f takes two parameters: z, the equilibrium, and x the model input.
        The output of f must be the same shape as z.
        :param alpha: step size for broyden's method
        """
        super(DEQ, self).__init__()
        self.f = f
        self.forward_eps = forward_eps
        self.RootFind = BroydenRootFind  # function where the magic happens
        self.alpha = alpha
        self.max_iters = max_iters
        self.eq_size = eq_size

    def forward(self, x):
        # TODO verify
        zstar = self.RootFind.apply(self.f, x, self.eq_size, self.forward_eps, self.alpha, self.max_iters)
        return zstar

