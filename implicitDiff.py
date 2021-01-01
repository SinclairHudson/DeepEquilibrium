import torch
import torch.autograd as AG
from rootfind import broyden

class ImplicitDiff(AG.Function):

    @staticmethod
    def forward(ctx, g, z_star):
        """
        :param g: the pytorch function that we've found the root to.
        :param z_star: the root of g, found arbitrarily.
        """

        ctx.originalg = g
        ctx.save_for_backward(z_star)
        return z_star

    @staticmethod
    def backward(ctx, dl_dzstar):
        """
        Input into this backward is the gradient of the loss wrt the equilibrium.
        From here, we want to pass a gradient to f, which in turn will pass it
        to the parameters within f. We can create this gradient however we want;
        we don't need torch ops, because we are the torch op.
        """
        z_star, = ctx.saved_tensors
        z_shape = z_star.size()
        # copy z_star, detach the copy from any graph,
        # and then enforce grad functionality
        z_star = z_star.reshape((-1,)).clone().detach().requires_grad_()

        with torch.enable_grad():
            # y and z_star, at this point, are both 1D vectors
            y = ctx.originalg(z_star)

        dl_dzstar_flat = dl_dzstar.reshape((-1,))
        # this function represents the LHS of eq 11 in the original paper
        # we use autograd to calculate the Jacobian-vector product
        def JacobianVector(x):
            y.backward(x, retain_graph=True)
            JTxT = z_star.grad.clone().detach()
            z_star.grad.zero_()  # remove the gradient (this is kind of a fake grad)
            return JTxT + dl_dzstar_flat

        neg_dl_dzs_J_inv, iters = broyden(JacobianVector, torch.zeros_like(z_star),
                                   2e-7, alpha=0.5, max_iters=200)

        wandb.log({"backward_iters": iters})

        return (None, neg_dl_dzs_J_inv.reshape(z_shape))
