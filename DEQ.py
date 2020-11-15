import torch.nn as nn
from rootfind import BroydenRootFind

class DEQ(nn.Module):
    def __init__(self, f, forward_eps, backward_eps, alpha):
        """
        :param forward_eps: epsilon
        """
        super(DEQ, self).__init__()
        self.f = f
        self.itercount = 0
        self.max_iterations = 150
        self.alpha = alpha
        self.alpha = alpha
        self.B = torch.eye(1)  # approximation of the inverse jacobian

    def g(z, x):
        return self.f(z, x) - z

    def iterate(self, z, x):
        self.itercount += 1
        g = self.g(z,x)
        z_next = z - alpha * torch.mv(self.B,g)  # eq 6 in original paper
        self.update_inv_jacobian_approx(z_next - z, self.g(z_next, x), - self.g(z, x))
        return z_next

    def update_inv_jacobian_approx(self, deltaZ, deltaG):
        """
        :param deltaZ: nx1 vector
        :param deltaG: nx1 vector
        """
        Bdg = torch.matmul(self.B, deltaG)  # nx1

        deltaZT = torch.transpose(deltaZ, 0, 1) # 1xn
        quotient = deltaZ - Bdg / torch.matmul(deltaZT, Bdg) # divided by constant,
        # quotient = nx1
        self.B = self.B - torch.matmul(quotient, torch.matmul(deltaZT, self.B))



    def forward(self, z_0, x):
        """
        z_0, and all following zs must be 1D vectors
        """
        self.itercount = 0
        equilibrium = False
        self.B = -1 * torch.eye(z_0.shape[0])  # n x n negative identity
        z_prev = z_0
        while not equilibrium:
            z_i = iterate(z_prev, x)
            equilibrium = torch.sum(torch.abs(z_i - z_prev)) < forward_eps:
            z_prev = z_i
        return z_prev, self.itercount

