#!/usr/bin/env python

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pandas as pd
import numpy as np
# from aaindex import MAXLEN

MAXLEN = 200

import sys

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def load_data(split=0.8):
    df = pd.read_pickle('data/data.pkl')
    df = df[df['indexes'].map(lambda x: len(x)) <= MAXLEN]
    n = len(df)
    print('Data size: {}'.format(n))
    index = np.arange(n)
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_n = int(n * split)
    train_df = df.iloc[index[:train_n]]
    test_df = df.iloc[index[train_n:]]

    def reshape(values):
        values = np.hstack(values).reshape(
            len(values), len(values[0]))
        return values

    def get_values(data_frame):
        n = len(data_frame)
        data = np.zeros((n, MAXLEN, 21), dtype=np.float32)
        for i, (_, row) in enumerate(data_frame.iterrows()):
            ind = row['indexes']
            m = len(row['indexes'])
            # s = (MAXLEN - m) // 2
            s = 0
            data[i, s: s + m, ind] = 1 
        return data.flatten()

    train_data = reshape(get_values(train_df))
    test_data = reshape(get_values(test_df))
    
    return train_data, test_data

train_data, test_data = load_data()
train_tensor = torch.from_numpy(train_data)
test_tensor = torch.from_numpy(test_data)
train_dataset = torch.utils.data.TensorDataset(train_tensor, train_tensor)
test_dataset = torch.utils.data.TensorDataset(test_tensor, test_tensor)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

        

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(MAXLEN * 21, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 800)
        self.fc41 = nn.Linear(800, 400)
        self.fc42 = nn.Linear(800, 400)
        self.fc5 = nn.Linear(400, 800)
        self.fc6 = nn.Linear(800, 800)
        self.fc7 = nn.Linear(800, 800)
        self.fc8 = nn.Linear(800, MAXLEN * 21)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        return self.fc41(h3), self.fc42(h3)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h4 = self.relu(self.fc5(z))
        h5 = self.relu(self.fc6(h4))
        h6 = self.relu(self.fc7(h5))
        return self.sigmoid(self.fc8(h6))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, MAXLEN))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
if args.cuda:
    model.cuda()


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, MAXLEN))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= args.batch_size * MAXLEN

    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            real = data[:n]
            decoded = recon_batch[:n]
            real = (real.cpu().data.numpy() * 8000).astype(np.int32)
            decoded = (decoded.cpu().data.numpy() * 8000).astype(np.int32)
            for j in range(n):
                print(real[j, :20])
                print(decoded[j, :20])
            
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    
