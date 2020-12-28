import time

import torch
import torch.nn as nn
import torch.optim as optim

from DEQ import DEQ
from unit import LinearUnit
import wandb

seed = 1337

torch.manual_seed(seed)
conf = {
    "epochs": 300_000,
    "pre_train_epochs": 2_000,
    "forward_eps": 1e-4,
    "max_iters": 150,
    "backward_eps": 1e-4,
    "alpha": 0.5,
    "learning_rate": 1e-4,
    "random_seed": seed,
}

wandb.init(project="deep-equilibrium", config=conf)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.MSELoss()

f = LinearUnit()

deq = DEQ(f, (-1, 2), conf["forward_eps"], conf["backward_eps"],
          conf["alpha"], conf["max_iters"])

deq_optim = optim.Adam(f.parameters(), lr=conf["learning_rate"])


for i in range(conf["pre_train_epochs"]):
    x = torch.rand((17, 2), requires_grad=True)
    z = torch.zeros((17, 2), requires_grad=True)
    y_hat = f(z, x)
    y_hat = f(y_hat, x)
    y_true = - x
    loss = criterion(y_true, y_hat)
    print(f"loss: {loss.item():.5f}")
    wandb.log({"loss": loss.item()})
    loss.backward()
    deq_optim.step()


for i in range(conf["epochs"]):
    x = torch.rand((17, 2), requires_grad=True)
    y_true = -x
    y_hat = deq.forward(x)
    loss = criterion(y_true, y_hat)
    print(f"deq_loss: {loss.item():.5f}")
    wandb.log({"loss": loss.item()})

    loss.backward()
    deq_optim.step()
