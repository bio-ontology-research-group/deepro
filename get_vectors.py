#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import math
from collections import Counter
from aaindex import is_ok
from utils import read_fasta, DataGenerator
from scipy import sparse
from keras.models import load_model


MAXLEN = 1000

@ck.command()
def main():
    run_model()

def load_data():
    ngram_df = pd.read_pickle('data/ngrams.pkl')
    vocab = {}
    for key, gram in enumerate(ngram_df['ngrams']):
        vocab[gram] = key + 1
    gram_len = len(ngram_df['ngrams'][0])
    print('Gram length:', gram_len)
    print('Vocabulary size:', len(vocab))

    ngrams = list()
    proteins = list()
    f = open('data/swissprot.fasta')
    prots, seqs = read_fasta(f.readlines())
    for protein, seq in zip(prots, seqs):
        if not is_ok(seq) or len(seq) - gram_len + 1 > MAXLEN:
            continue
        proteins.append(protein)
        grams = list()
        for i in range(len(seq) - gram_len + 1):
            grams.append(vocab[seq[i: (i + gram_len)]])
        ngrams.append(grams)
        
    df = pd.DataFrame({
        'proteins': proteins,
        'ngrams': ngrams,
    })

    def get_values(df):
        grows = []
        gcols = []
        gdata = []
        for i, row in enumerate(df.itertuples()):
            for j in range(len(row.ngrams)):
                grows.append(i)
                gcols.append(j)
                gdata.append(row.ngrams[j])
        data = sparse.csr_matrix((gdata, (grows, gcols)), shape=(len(df), MAXLEN))
        return data

    return proteins, get_values(df)

def run_model(model_path='data/model.h5', batch_size=128):
    model = load_model(model_path)
    prots, data = load_data()
    data_generator = DataGenerator(batch_size)
    data_generator.fit(data)
    # Features layer model
    model = model.layers[1]
    steps = math.ceil(data.shape[0] / batch_size)
    output = model.predict_generator(data_generator, steps=steps, verbose=1)
    print(output)
    vectors = list()
    for i in range(output.shape[0]):
        vectors.append(output[i, :])
    df = pd.DataFrame({'proteins': prots, 'vectors': vectors})
    df.to_pickle('data/vectors.pkl')

if __name__ == '__main__':
    main()
