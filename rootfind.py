import torch.autograd as AG


class BroydenRootFind(AG.Function):
    """
    This is the function that is actually used to solve a rootfinding problem.
    It inherits from autograd.Function, so it can be backpropped through.
    """
    @staticmethod
    def update_inv_jacobian_approx(B, deltaZ, deltaG):
        """
        :param B: nxn inv jacobian approx
        :param deltaZ: nx1 vector
        :param deltaG: nx1 vector
        """
        Bdg = torch.matmul(B, deltaG)  # nx1

        deltaZT = torch.transpose(deltaZ, 0, 1) # 1xn
        quotient = deltaZ - Bdg / torch.matmul(deltaZT, Bdg) # divided by constant,
        # quotient = nx1
        nextB = B - torch.matmul(quotient, torch.matmul(deltaZT, self.B))
        return nextB

    @staticmethod
    def iterate(g, z, x):
        return g(z,x)

    @staticmethod
    def forward(ctx, g, z_0, x, eps):
        """
        :param func:
        :returns: equilibrium z_star
        """

        iters = 0
        while delta < eps and iters < 150:


    @staticmethod
    def backward(ctx, dl_dz, ):
