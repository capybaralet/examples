from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import copy
import numpy as np


"""
TODO:
    set-up saving
        save-path (FIXME!)
        results (just learning curves?)
    port to cluster and run with different thresholds
        git
        make sure it runs
        launcher
            figure out launch str
    set-up plotting (look at some plots!?)
    PROFIT!
"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                    help='SGD momentum (default: 0.0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
########### DK added (below)
#parser.add_argument('--setting', type=str, default="default")
parser.add_argument('--improvement_threshold', type=float, default=0.) # how much do we need to improve by, in order to accept an update?
args = parser.parse_args()

PATH = 'improvement_threshold=' + str(args.improvement_threshold) + '____'

use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# so bad... I should be using valid set!
tr_loss = []
va_loss = []
tr_acc = []
va_acc = []

te_loss = []
te_acc = []

for epoch in range(1, args.epochs + 1): # we'll just use the first half of the data at each epoch...

    #def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    print ("begin training epoch", str(epoch))
    etl = enumerate(train_loader)

    loss = 0
    for i in range(len(train_loader) // 2):
        #print (i)
        # save current parameters:
        params = copy.deepcopy(model.state_dict())
        # TODO: how am I SUPPOSED to do this? 
        # TODO: also monitor (and compare difference in loss on the CURRENT mini-batch)
        tr_data, tr_target = etl.__next__()[1]
        gen_data, gen_target = etl.__next__()[1]
        # check how well the current model generalizes
        output = model(gen_data)
        gen_loss_pre_update = F.nll_loss(output, gen_target)
        # update the parameters
        optimizer.zero_grad()
        output = model(tr_data)
        tr_loss = loss
        loss = F.nll_loss(output, tr_target)
        tr_diff = tr_loss - loss
        loss.backward()
        optimizer.step()
        # check whether the update lead to enough improvement:
        output = model(gen_data)
        gen_loss_post_update = F.nll_loss(output, gen_target)
        gen_diff = gen_loss_pre_update - gen_loss_post_update 
        if gen_diff < args.improvement_threshold: # LARGER is better
            # undo the last update
            model.load_state_dict(params)
            print ("\t\t\t\t\t\tupdate REJECTED, gen_diff="  + str(np.round(gen_diff.detach().numpy()[()], 5)) + "  tr_diff=" + str(tr_diff.detach().numpy()[()]))
        else:
            print ("\t\t\t\t\t\tupdate accepted, gen_diff="  + str(np.round(gen_diff.detach().numpy()[()], 5)) + "  tr_diff=" + str(tr_diff.detach().numpy()[()]))


        batch_idx = i
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(tr_data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    #def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    te_loss.append(test_loss)
    te_acc.append(acc)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))

    np.savetxt(PATH + 'te_loss', te_loss)
    np.savetxt(PATH + 'te_acc', te_acc)

if (args.save_model):
    torch.save(model.state_dict(),"mnist_cnn.pt")



