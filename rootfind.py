import torch
import torch.autograd as AG
import wandb


def update_inv_jacobian_approx(B, deltaZ, deltaG):
    """
    This is eq 10 in the paper
    :param B: nxn inv jacobian approx
    :param deltaZ: n vector
    :param deltaG: n vector
    """
    Bdg = torch.mv(B, deltaG)  # nx1

    rational = deltaZ - Bdg
    rational = rational / (torch.dot(deltaZ, Bdg) + 1e-10)

    deltaZT = torch.unsqueeze(deltaZ, 0)  # 1xn
    notrational = torch.matmul(deltaZT, B)  # 1xn
    rational = torch.unsqueeze(rational, 1)  # nx1
    update = torch.matmul(rational, notrational)
    nextB = B + update  # n x n
    return nextB


def broyden(g, z_0, eps, alpha, max_iters):
    """
    This function performes broyden's method, finding the root of g.
    This function is used by both the forward and backward pass.
    :param z_0: the first estimate of a root
    :param eps: the error tolerance of an accepted root.
    :param alpha: the step size
    :param max_iters: the maximum number of broyden steps
    """
    prevZ = z_0
    prevG = g(z_0)

    B = - torch.eye(z_0.size()[0])

    delta = float("inf")  # arbitrarily large, first diff is undefined
    iters = 0
    # while the error is larger than epsilon, and we're not at the max_iters
    while eps < delta and iters < max_iters:
        iters += 1
        newZ = prevZ - alpha * torch.matmul(B, prevG)  # eq 6 in the paper
        newG = g(newZ)
        deltaG = newG - prevG
        deltaZ = newZ - prevZ
        delta = torch.norm(deltaZ)

        # update the jacob approx
        B = update_inv_jacobian_approx(B, deltaZ, deltaG)
        prevG = newG  # the new is now the old
        prevZ = newZ

    return prevZ, iters


class BroydenRootFind(AG.Function):
    """
    This is the function that is actually used to solve a rootfinding problem.
    It inherits from autograd.Function, so it can be backpropped through.
    Can we make this non-abstract?
    """

    @staticmethod
    def forward(ctx, g, z_0, eps, alpha, max_iters):
        """
        :param func:
        :returns: equilibrium z_star, the root of f(z,x)
        """
        root, iters = broyden(g, z_0, eps, alpha, max_iters)

        wandb.log({"forward_iters": iters})
        return root

    @staticmethod
    def backward(ctx, dl_dzstar):
        """
        This method does nothing. while we have dl_dzstar, we don't want to
        apply the gradient
        """
        raise NotImplementedError("Backpropping through a general Broyden's \
                                  method is not defined. Use ImplicitDiff.")
