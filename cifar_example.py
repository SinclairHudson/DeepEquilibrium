import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from DEQ import DEQ
import wandb

seed = 1337

torch.manual_seed(seed)
conf = {
    "epochs": 30,
    "pre_train_epochs": 3,
    "forward_eps": 1e-4,
    "max_iters": 150,
    "backward_eps": 1e-4,
    "batch_size": 4,
    "alpha": 0.5,
    "learning_rate": 1e-4,
    "random_seed": seed,
}

wandb.init(project="deep-equilibrium", config=conf)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("using gpu")
# the following line makes sure that all initialized tensors are made on gpu
torch.cuda.manual_seed_all(seed)



class Unit(nn.Module):
    def __init__(self):
        super(Unit, self).__init__()
        self.conv1 = nn.Conv2d(8, 6, 5, padding=2)  # need same size
        self.conv2 = nn.Conv2d(6, 5, 5, padding=2)
        self.gn = nn.GroupNorm(num_groups=3, num_channels=6)

    def forward(self, z, x):
        """
        z will be Nx5x32x32, x is Nx3x32x32
        """
        x = torch.cat((z, x), dim=1)
        x = self.gn(F.leaky_relu(self.conv1(x)))
        x = F.leaky_relu(self.conv2(x))
        return x


u = Unit()

detectionHead = nn.Conv2d(5, 10, 32, padding=0)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.deq = DEQ(u, (-1, 5, 32, 32), conf["forward_eps"],
                       conf["backward_eps"], conf["alpha"],
                       conf["max_iters"])
        # covers the whole image:
        self.class_head = detectionHead

    def forward(self, image):
        x = self.deq(image)
        x = self.class_head(x)
        return x


# data

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=conf["batch_size"],
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=conf["batch_size"],
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# losses and optimization
criterion = nn.CrossEntropyLoss().to(device)

deq_classifier = Classifier().to(device)

deq_optim = optim.Adam(u.parameters(), lr=conf["learning_rate"])

print("starting pretraining")
for e in range(conf["pre_train_epochs"]):
    for i, data in enumerate(trainloader, 0):
        image, labels = data

        deq_optim.zero_grad()

        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        f_start = time.time()
        im = image.to(device)
        z = torch.zeros((conf["batch_size"], 5, 32, 32))

        y_hat = u(z, im)
        y_hat = u(y_hat, im)
        y_hat = detectionHead(y_hat)
        f_end = time.time()

        loss = criterion(y_hat.squeeze(), labels.to(device))
        b_start = time.time()
        loss.backward()
        b_end = time.time()
        deq_optim.step()
        print(f"pretraining_loss: {loss.item():.5f} | epoch: {e} | batch: {i}")
        wandb.log({"loss": loss.cpu().item(),
                   "forward pass runtime": f_end - f_start,
                   "backward pass runtime": b_end - b_start})

    torch.set_default_tensor_type('torch.FloatTensor')

for e in range(conf["epochs"]):
    for i, data in enumerate(trainloader, 0):
        image, labels = data

        # reset the gradients
        deq_optim.zero_grad()

        # the DEQ forward pass creates tensors, and we want them to be created
        # on a GPU instead of a CPU
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        f_start = time.time()
        y_hat = deq_classifier(image.to(device))
        f_end = time.time()

        loss = criterion(y_hat.squeeze(), labels.to(device))
        b_start = time.time()
        loss.backward()
        b_end = time.time()
        deq_optim.step()
        print(f"deq_loss: {loss.item():.5f} | epoch: {e} | batch: {i}")
        wandb.log({"loss": loss.cpu().item(),
                   "forward pass runtime": f_end - f_start,
                   "backward pass runtime": b_end - b_start})

    # reset because the trainloader wants to be on CPU
    torch.set_default_tensor_type('torch.FloatTensor')
