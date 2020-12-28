import torch
import torch.nn as nn
from rootfind import BroydenRootFind
from rootfind import ImplicitDiff


class DEQ(nn.Module):
    """
    A module to represent a DEQ.
    """
    def __init__(self, f, eq_shape, forward_eps, backward_eps, alpha, max_iters):
        """
        :param forward_eps: epsilon for the equilibrium in the forward pass
        :param backward_eps: epsilon for the equilibrium in the backward pass
        :param f: The Pytorch differentiable function that will be "iterated".
        This could be a nn.Module, or a Functional layer. Should have parameters
        f takes two parameters: z, the equilibrium, and x the model input.
        The output of f must be the same shape as z.
        f can take any shape of input, but broyden's method will be applied
        per-batch. That is, broyden's method only terminates once all examples
        in the batch have converged.
        :param alpha: step size for broyden's method
        """
        super(DEQ, self).__init__()
        self.f = f
        self.forward_eps = forward_eps
        self.RootFind = BroydenRootFind
        self.alpha = alpha
        self.max_iters = max_iters
        self.eq_shape = list(eq_shape)

    def forward(self, x):
        """
        :param x: Tensor with unspecified size, however the first dimension is
        a batch dimension
        """
        batch_size = x.size()[0]
        self.eq_shape[0] = batch_size

        def g(z):
            """
            This is the function we will find the root of. The root is equal
            to the equilibrium of f.
            :param z_i: a vector in the equilibrium space (1D)
            :returns z_{i+1}: a vector in the equilibrium space (1D)
            """
            if len(z.size()) != 1:
                raise ValueError(f"g only takes 1D tensors, but a tensor of \
                                 {z.size()} was provided.")
            z_reshaped = torch.reshape(z, self.eq_shape)
            result = self.f(z_reshaped, x) - z_reshaped
            result_flat = torch.reshape(result, (-1,))
            return result_flat

        with torch.no_grad():
            z_0 = torch.zeros(self.eq_shape).reshape((-1,))
            z_star_flat = self.RootFind.apply(g, z_0, self.forward_eps,
                                         self.alpha, self.max_iters)

        # this is a dummy call, to pass the gradient to the parameters of f.
        z_star = torch.reshape(z_star_flat, self.eq_shape)
        z_star = self.f(z_star, x)
        # this call doesn't modify z_star, but ensures we differentiate
        # properly in the backward pass.
        z_star = ImplicitDiff.apply(g, z_star)

        return z_star

