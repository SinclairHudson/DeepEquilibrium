# Deep Equilibrium Models
Repository implementing [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377).

If you use this repository, be sure to cite the authors of the original paper.
This implementation is mine, but the ideas are theirs.

Also check out [their implementation](https://github.com/locuslab/deq).

## What's a Deep Equilibrium Model?

DEQs are a form of implicit deep learning; they don't simply move data through
layers and end up at a result.

Read the original paper. Also, the same authors provide a more concise and general
formulation of DEQs in their follow-up work, on [Multiscale Deep Equilibrium Models](https://arxiv.org/abs/2006.08656).


## TODO List:
- [x] Trivial example working, in the forward pass (size 10)
- [x] Trivial example working in the backwards pass (learn identity or constant)
- [x] Get batching working (how does original repo do it?)
- [x] MNIST (working?)
- [ ] Use the approximate Jacobian Inverse and not the full $n \times n$ matrix
- [x] More complicated 2D and 3D inputs and outputs

## Contributing:
Anyone is welcome to open an issue or PR. I think my implementation is correct,
but if you see something let me know! I found the original paper really interesting,
but the official repo very confusing.
