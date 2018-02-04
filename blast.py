#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import math
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity

MAXLEN = 1000

@ck.command()
def main():
    res = load_sw_scores()
    blast = list()
    df = pd.read_pickle('data/vectors.pkl')
    proteins = df['proteins'].values
    vectors = df['vectors'].values
    vec_matrix = np.empty((len(vectors), len(vectors[0])), dtype=np.float32)
    for i in range(len(vectors)):
        vec_matrix[i, :] = vectors[i]
    cosine_sim = cosine_similarity(vec_matrix)
    cosine = list()
    for i in range(len(proteins)):
        p1 = proteins[i]
        for j in range(i + 1, len(proteins)):
            cosine.append(cosine_sim[i, j])
            p2 = proteins[j]
            if p1 in res and p2 in res[p1]:
                blast.append(res[p1][p2] / res[p1][p1])
            else:
                blast.append(0.0)
    print(spearmanr(cosine, blast))
    print(pearsonr(cosine, blast))

def load_blast_sim():
    res = {}
    with open('data/sim.blst') as f:
        for line in f:
            it = line.strip().split()
            p1 = it[0]
            p2 = it[1]
            s = float(it[2])
            if p1 not in res:
                res[p1] = {}
            if p2 not in res:
                res[p2] = {}
            res[p1][p2] = s
            res[p2][p1] = s
    return res

def load_sw_scores():
    res = {}
    with open('data/scores.sw') as f:
        for line in f:
            line = line.strip()
            if line.startswith('query:'):
                cur_prot = line[6:]
                res[cur_prot] = {}
            elif line.startswith('score: '):
                it = line[7:].split(' -- ')
                res[cur_prot][it[1]] = int(it[0])
    return res
    

if __name__ == '__main__':
    main()
