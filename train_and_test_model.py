from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np


save_directory = 'saved_data'

parser = argparse.ArgumentParser(description='PyTorch toy Example')


parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

use_both_proba_and_target = False

data = torch.from_numpy(np.load('%s/X_train.npy' % save_directory).astype(float)).float()
proba = torch.from_numpy(np.load('%s/Y_train.npy' % save_directory).astype(float)).float()

train_batch_size = 1000
test_batch_size = 1000
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, proba),
    batch_size=train_batch_size, shuffle=True, **kwargs)



proba = np.load('%s/Y_test.npy' % save_directory).astype(float)
proba = torch.from_numpy(proba).float()
test_data = torch.from_numpy(np.load('%s/X_test.npy' % save_directory).astype(float)).float()


test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data, proba),
    batch_size=test_batch_size, shuffle=False, **kwargs)

nb_input_dimensions = 3
nb_output_dimensions = nb_input_dimensions
nb_hidden_dimensions = 30

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(nb_input_dimensions, nb_hidden_dimensions)
        self.hidden2 = nn.Linear(nb_hidden_dimensions, nb_hidden_dimensions)
        self.out   = nn.Linear(nb_hidden_dimensions,nb_output_dimensions)

    def forward(self, x):
        x = F.tanh(self.hidden(x))
        x = F.tanh(self.hidden2(x))
        x = self.out(x)
        return x
model = Net()


class ClusterEmbedding(nn.Module):
    def __init__(self, y_target):
        super(ClusterEmbedding, self).__init__()

        self.n_examples = y_target.size(0)
        
        self.inds = Variable(torch.arange(0, self.n_examples).long())
        self.y_target = Variable(y_target)
        
        self.embedding = nn.Embedding(self.n_examples, 2)

    def forward(self):
        return self.embedding.forward(self.inds)
    



if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.0001)


def inv_H(H_prev):
    H_prev_inv = H_prev.t()
    d = H_prev_inv.sum(1).unsqueeze(1).expand_as(H_prev_inv)
    d[d==0] = 1
    return H_prev_inv / d

def pseudo_inverse(X):
    u, s, v = torch.svd(X)
    h = torch.max(s) * float(max(X.size(0),X.size(1))) * 1e-15
    indices = torch.ge(s,h)
    indices2 = indices.eq(0)
    s[indices] = 1.0 / s[indices]
    s[indices2] = 0
    return torch.mm(torch.mm(v, torch.diag(s)), u.t())

 

def grad_F(F, H):
    inv_F = pseudo_inverse(F)
    return torch.mm(torch.mm(F,torch.mm(inv_F,H)) - H,torch.mm(inv_H(H),inv_F.t()))



def spectral_learning(epoch):
    model.train()
    enum_train = enumerate(train_loader)
    for batch_idx, (data, Y) in enum_train:
        if args.cuda:
            data, Y = data.cuda(), Y.cuda()
        data, Y = Variable(data), Variable(Y)
        optimizer.zero_grad()
        F = model.forward(data)
        G = grad_F(F,Y)
        F.backward(gradient=G)
        optimizer.step()
        objective_value = Y.size()[1] - torch.sum(torch.mm(pseudo_inverse(Y),F)  * torch.mm(pseudo_inverse(F), Y).t())

        print("epoch %d --- loss value= %f" % (epoch, objective_value))



def save_to_file(iteration):
    model.eval()

    fdata = open("%s/test_input_data.txt"  % save_directory,"w")
    foutput = open("%s/test_output_data.txt"  % save_directory,"w")
    flabels = open("%s/test_labels.txt"  % save_directory,"w")

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        target = target.max(1)[1] + 1
        z = model.forward(data)
        for i in range(z.size()[0]):
            for j in range(nb_input_dimensions):
                if args.cuda:
                    fdata.write("%f " % data[i][j].data.cpu().numpy().astype(float))
                else:
                    fdata.write("%f " % data[i][j].data.numpy().astype(float))
            fdata.write("\n")
            for j in range(nb_output_dimensions):
                if args.cuda:
                    foutput.write("%f " % z[i][j].data.cpu().numpy().astype(float))
                else:
                    foutput.write("%f " % z[i][j].data.numpy().astype(float))
            foutput.write("\n")
            if args.cuda:
                flabels.write("%d\n" % target[i].data.cpu().numpy().astype(int))
            else:
                flabels.write("%d\n" % target[i].data.numpy().astype(int))
        
    fdata.close()
    flabels.close()
    foutput.close()

nb_epochs = 100

####### training

print("Starting training")
for epoch in range(1, nb_epochs+1):
    spectral_learning(epoch)    

print("Training complete")

####### saving test representations

print("Saving test representations")
save_to_file(nb_epochs)

print("Test representations saved")

