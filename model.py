#!/usr/bin/env python

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import click as ck
import pandas as pd


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, ipro_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.ipro_embeddings = nn.Embedding(ipro_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        lstm_out = lstm_out[len(sentence) - 1]
        tag_space = self.hidden2tag(lstm_out.view(1, -1))
        tag_scores = F.softmax(tag_space)
        return tag_scores

    

@ck.command()
def main():
    train_model()


def load_data(split=0.8):
    data = pd.read_pickle('data/data.pkl')
    n = len(data)
    index = np.arange(int(n / 100))
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_n = int(n * split)
    valid_n = int(train_n * split)
    train = data.iloc[index[:valid_n]]
    valid = data.iloc[index[valid_n:train_n]]
    test = data.iloc[index[train_n:]]
    
    interpros = pd.read_pickle('data/dictionary.pkl')['interpros'].values
    class_ix = {}
    for i, ipro in enumerate(interpros):
        class_ix[ipro] = i
    return train, valid, test, class_ix

def train_model(embedding_dim=8, hidden_dim=8, epochs=12):
    train, valid, test, class_ix = load_data()
    # Number of all possible trigrams
    vocab_size = 8001
    class_size = len(class_ix)
    
    model = LSTMTagger(embedding_dim, hidden_dim, vocab_size, class_size)
    model = model.cuda()
    loss_function = nn.MultiLabelMarginLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)

    train_data = train[['ngrams', 'interpros']].values
    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))
        with ck.progressbar(train_data) as data:
            train_loss = 0.0
            for item in data:
                # Clear gradients
                model.zero_grad()
                # Clear hidden state for each instance
                model.hidden = model.init_hidden()
                
                inputs_tensor = torch.LongTensor(item[0]).cuda()
                inputs = autograd.Variable(inputs_tensor).cuda()
                labels = []
                for ipro in item[1]:
                    if ipro in class_ix:
                        labels.append(class_ix[ipro])
                if len(labels) == 0:
                    continue
                labels = autograd.Variable(torch.LongTensor(labels)).cuda()
                scores = model(inputs)
                loss = loss_function(scores, labels)
                train_loss += loss
                loss.backward()
                optimizer.step()
            print('Training Loss: ', train_loss / len(train_data))
                
                
if __name__ == '__main__':
    main()
