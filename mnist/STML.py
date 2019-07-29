from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# DK
from pylab import *

import copy
import os
import argparse
import shutil
import sys
import numpy 
np = numpy

import time



"""
TODO:
    set-up saving
        save-path (FIXME!)
        results (just learning curves?)
    port to cluster and run with improvementerent thresholds
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
parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
########### DK added (below)
#parser.add_argument('--setting', type=str, default="default")
parser.add_argument('--improvement_threshold', type=float, default=0.) # how much do we need to improve by, in order to accept an update?
parser.add_argument('--n_seeds', type=int, default=20)
parser.add_argument('--save_dir', type=str, default=os.environ['SCRATCH']) # N.B.! you must specify the environment variable SCRATCH.  you can do this like: export $SCRATCH=<<complete file-path for the save_dir>>
parser.add_argument('--data_path', type=str, default=os.environ['SLURM_TMPDIR']) # N.B.! you must specify the environment variable SCRATCH.  you can do this like: export $SCRATCH=<<complete file-path for the save_dir>>

############################################################################333
args = parser.parse_args()
print (args)
args_dict = args.__dict__

# TODO: why do I end up with single quotes around the directory name?
if args_dict['save_dir'] is None:
    try:
        save_dir = os.environ['SCRATCH']
    except:
        print ("\n\n\n\t\t\t\t WARNING: save_dir is None! Results will not be saved! \n\n\n")
else:
    # save_dir = filename + PROVIDED parser arguments
    flags = [flag.lstrip('--') for flag in sys.argv[1:] if not (flag.startswith('--save_dir') or flag.startswith('--train'))]
    exp_title = '_'.join(flags)
    save_dir = os.path.join(args_dict.pop('save_dir'), os.path.basename(__file__) + '___' + exp_title)
    print("\t\t save_dir=",  save_dir)

    # make directory for results
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # save a copy of THIS SCRIPT in save_dir
    shutil.copy(__file__, os.path.join(save_dir,'exp_script.py'))
    # save ALL parser arguments
    with open (os.path.join(save_dir,'exp_settings.txt'), 'w') as f:
        for key in sorted(args_dict):
            f.write(key+'\t'+str(args_dict[key])+'\n')

locals().update(args_dict)
############################################################################333

PATH = 'improvement_threshold=' + str(improvement_threshold) + '____'

args = parser.parse_args()

###################################

use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_path, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_path, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)















threshs = [-np.inf, -.1, -.01, -.001, 0, .001, .01, .1]

assert args.batch_size == 64
n_batches = 460
n_steps =  n_batches * args.epochs

tr_improvements = np.inf * np.ones((n_seeds, len(threshs), n_steps))
gen_improvements = np.inf * np.ones((n_seeds, len(threshs), n_steps))
#
tr_loss = np.inf * np.ones((n_seeds, len(threshs), args.epochs))
tr_acc = np.inf * np.ones((n_seeds, len(threshs), args.epochs))
gen_loss = np.inf * np.ones((n_seeds, len(threshs), args.epochs))
gen_acc = np.inf * np.ones((n_seeds, len(threshs), args.epochs))
# FIXME: so bad... I should be using valid set!
te_loss = np.inf * np.ones((n_seeds, len(threshs), args.epochs))
te_acc = np.inf * np.ones((n_seeds, len(threshs), args.epochs))



n_experiments = n_seeds * len(threshs)
t0 = time.time()

for seed in range(n_seeds):
    for thresh_n, thresh in enumerate(threshs):

        experiment_n = seed * len(threshs) + thresh_n
        print ('\n\n')
        print ('\n\n')
        print ("experiment #", experiment_n, " out of ", n_experiments)
        print ("total progress: ", np.round(100 * experiment_n / n_experiments), "%")
        print ("total time: ", time.time() - t0)
        print ('\n\n')
        print ('\n\n')

        step = 0

        for epoch in range(1, args.epochs + 1): # we'll just use the first half of the data at each epoch...
            print ("seed, thresh_n, epoch = ", seed, thresh_n, epoch)
            print ("total time: ", time.time() - t0)

            # TRAIN
            model.train()
            print ("begin training epoch", str(epoch))
            loss = 0
            etl = enumerate(train_loader)
            for i in range(len(train_loader) // 2):
                #print (i)
                # save current parameters:
                params = copy.deepcopy(model.state_dict())
                # TODO: how am I SUPPOSED to do this? 
                tr_data, tr_target = etl.__next__()[1]
                gen_data, gen_target = etl.__next__()[1]
                tr_data, tr_target = tr_data.to(device), tr_target.to(device)
                gen_data, gen_target = gen_data.to(device), gen_target.to(device)
                # check how well the current model generalizes
                output = model(gen_data)
                gen_loss_pre_update = F.nll_loss(output, gen_target)
                # update the parameters
                optimizer.zero_grad()
                output = model(tr_data)
                old_loss = loss
                loss = F.nll_loss(output, tr_target)
                tr_improvement = old_loss - loss
                loss.backward()
                optimizer.step()
                # check whether the update lead to enough improvement:
                output = model(gen_data)
                gen_loss_post_update = F.nll_loss(output, gen_target)
                gen_improvement = gen_loss_pre_update - gen_loss_post_update 
                tr_improvement = tr_improvement.detach().numpy()[()]
                gen_improvement = gen_improvement.detach().numpy()[()]
                tr_improvements[seed, thresh_n, step] = tr_improvement
                gen_improvements[seed, thresh_n, step] = gen_improvement

                if gen_improvement > args.improvement_threshold:
                    print ("\t\t\t\t\t\tupdate accepted, gen_improvement="  + str(np.round(gen_improvement, 4)) + "  tr_improvement=" + str(np.round(tr_improvement, 4)))
                else:
                    # undo the last update
                    model.load_state_dict(params)
                    print ("\t\t\t\t\t\tupdate REJECTED, gen_improvement="  + str(np.round(gen_improvement, 4)) + "  tr_improvement=" + str(np.round(tr_improvement, 4)))


                batch_idx = i
                if batch_idx % args.log_interval == 0: # TODO: misleading...
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(tr_data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                step = step + 1

            # TEST (evaluate test loss and accuracy)
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
            test_acc = 100. * correct / len(test_loader.dataset)
            te_loss[seed, thresh_n, epoch] = test_loss
            te_acc[seed, thresh_n, epoch] = test_acc

            # TEST (evaluate train / gen loss and accuracy)
            model.eval()
            train_loss = 0
            train_correct = 0
            generalization_loss = 0
            generalization_correct = 0
            #
            etl = enumerate(train_loader)
            with torch.no_grad():
                for i in range(len(train_loader) // 2):
                    # TR
                    data, target = etl.__next__()[1]
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    train_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                    train_correct += pred.eq(target.view_as(pred)).sum().item()
                    # GEN
                    data, target = etl.__next__()[1]
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    generalization_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                    generalization_correct += pred.eq(target.view_as(pred)).sum().item()
            # TODO
            train_loss /= n_batches
            train_acc = 100. * correct / n_batches
            tr_loss[seed, thresh_n, epoch] = train_loss
            tr_acc[seed, thresh_n, epoch] = train_acc
            #
            generalization_loss /= n_batches
            generalization_acc = 100. * correct / n_batches
            gen_loss[seed, thresh_n, epoch] = generalization_loss
            gen_acc[seed, thresh_n, epoch] = generalization_acc




            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset), test_acc))

            np.save(os.path.join(save_dir, 'tr_improvements.npy'), tr_improvements)
            np.save(os.path.join(save_dir, 'gen_improvements.npy'), gen_improvements)

            np.save(os.path.join(save_dir, 'tr_loss.npy'), tr_loss)
            np.save(os.path.join(save_dir, 'tr_acc.npy'), tr_acc)
            np.save(os.path.join(save_dir, 'gen_loss.npy'), gen_loss)
            np.save(os.path.join(save_dir, 'gen_acc.npy'), gen_acc)
            np.save(os.path.join(save_dir, 'te_loss.npy'), te_loss)
            np.save(os.path.join(save_dir, 'te_acc.npy'), te_acc)


            # TODO: rm / smoothing
            if 0:
                figure()
                plot(tr_improvements, label='tr_improvements')
                plot(gen_improvements, label='gen_improvements')
                legend()

