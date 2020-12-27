import torch
import torch.autograd as AG


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

    B = - torch.eye(2)

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

        B = update_inv_jacobian_approx(
            B, deltaZ, deltaG)  # update the jacob approx
        prevG = newG  # the new is now the old
        prevZ = newZ

    return prevZ


class BroydenRootFind(AG.Function):
    """
    This is the function that is actually used to solve a rootfinding problem.
    It inherits from autograd.Function, so it can be backpropped through.
    Can we make this non-abstract?
    """

    @staticmethod
    def forward(ctx, f, x, eq_size, eps, alpha, max_iters):
        """
        :param func:
        :returns: equilibrium z_star, the root of f(z,x)
        """

        def g(z):
            return f(z, x) - z

        # TODO is this a sin?
        # saving for backward
        ctx.originalg = g

        with torch.no_grad():
            root = broyden(g, torch.zeros((eq_size,)), eps, alpha, max_iters)

        ctx.save_for_backward(root)  # z_star
        return root

    @staticmethod
    def backward(ctx, dl_dzstar):
        """
        Input into this backward is the gradient of the loss wrt the equilibrium
        From here, we want to pass a gradient to f, which in turn will pass it
        to the parameters within f. We can create this gradient however we want;
        we don't need torch ops, because we are the torch op.
        """
        z_star, = ctx.saved_tensors
        # copy z_star, detach the copy from any graph, and then enforce grad functionality
        z_star = z_star.clone().detach().requires_grad_()

        with torch.enable_grad():
            y = ctx.originalg(z_star)

        # this function represents the LHS of eq 11 in the original paper
        # we use autograd to calculate the Jacobian-vector product
        def JacobianVector(x):
            y.backward(x, retain_graph=True)
            JTxT = z_star.grad.clone().detach()
            z_star.grad.zero_()  # remove the gradient (this is kind of a fake grad)
            return JTxT + dl_dzstar

        neg_dl_dzs_J_inv = broyden(JacobianVector, torch.zeros_like(dl_dzstar),
                                   2e-6, alpha=1, max_iters=200)

        return (None, neg_dl_dzs_J_inv, None, None, None, None)
