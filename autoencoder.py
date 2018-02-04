#!/usr/bin/env python

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import click as ck
import pandas as pd

SOS_token = 0
EOS_token = 8001

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



@ck.command()
@ck.option('--cuda', is_flag=True)
def main(cuda):
    global use_cuda
    use_cuda = cuda
    train_model()


def load_data(split=0.8):
    data = pd.read_pickle('data/data.pkl')
    n = len(data)
    index = np.arange(n)
    np.random.seed(seed=0)
    np.random.shuffle(index)
    data = data.iloc[index]
    ngrams = data['ngrams'].values
    for i in range(len(ngrams)):
        ngrams[i].append(EOS_token)
        
    return ngrams


def train_model(embedding_dim=128, hidden_dim=128, epochs=12):
    train_data = load_data()[:1000]
    # Number of all possible trigrams
    vocab_size = 8002
    
    encoder = EncoderRNN(vocab_size, embedding_dim)
    decoder = DecoderRNN(embedding_dim, vocab_size)
    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        
    loss_function = nn.NLLLoss()
    encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=0.01)
    decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=0.01)

    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))
        with ck.progressbar(train_data) as data:
            train_loss = 0.0
            for item in data:
                # Clear gradients
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                # Clear hidden state for each instance
                enc_hidden = encoder.init_hidden()

                seq_length = len(item)
                encoder_outputs = Variable(torch.zeros(seq_length, encoder.hidden_size))
                encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
                inputs = Variable(torch.LongTensor(item))
                inputs = inputs.cuda() if use_cuda else inputs
                for i in range(seq_length):
                    enc_out, enc_hidden = encoder(inputs[i], enc_hidden)
                    encoder_outputs[i] = enc_out[0][0]
                dec_input = Variable(torch.LongTensor([[SOS_token]]))
                dec_input = dec_input.cuda() if use_cuda else dec_input
                dec_hidden = enc_hidden
                loss = 0.0
                for i in range(seq_length):
                    dec_output, dec_hidden = decoder(dec_input, dec_hidden)
                    loss += loss_function(dec_output, inputs[i])
                    dec_input = inputs[i]
                loss /= seq_length
                train_loss += loss
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
            print('Training Loss: ', train_loss / len(train_data))
                
                
if __name__ == '__main__':
    main()
