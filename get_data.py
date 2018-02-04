#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
from collections import Counter
from aaindex import is_ok, AAINDEX
from utils import read_fasta


@ck.command()
def main():
    to_pandas()
    # get_sw_scores()


def get_sw_scores():
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
    p = list(res.keys())
    s = list()
    for i in range(len(p)):
        scores = np.zeros((len(p), ), dtype=np.float32)
        for j in range(len(p)):
            norm = max(res[p[i]][p[i]], res[p[j]][p[j]])
            scores[j] = res[p[i]][p[j]] / norm
        s.append(scores)

    df = pd.DataFrame({'proteins': p, 'scores': s})

    prots, sequences = read_fasta(open('data/swissprot.fasta', 'r'))
    prots_dict = {}
    for prot, seq in zip(prots, sequences):
        prots_dict[prot] = seq

    sequences = list()
    for prot in p:
        sequences.append(prots_dict[prot])
    df['sequences'] = sequences

    df.to_pickle('data/sw_scores.pkl')
            
    return res


def to_pandas():
    ngram_df = pd.read_pickle('data/ngrams.pkl')
    vocab = {}
    for key, gram in enumerate(ngram_df['ngrams']):
        vocab[gram] = key + 1
    gram_len = len(ngram_df['ngrams'][0])
    print('Gram length:', gram_len)
    print('Vocabulary size:', len(vocab))

    proteins = list()
    accessions = list()
    sequences = list()
    interpros = list()
    ngrams = list()
    indexes = list()
    counter = Counter()
    maxlen = 0
    with open('data/data.tsv') as f:
        for line in f:
            items = line.strip().split('\t')
            seq = items[2]
            if not is_ok(seq) or len(seq) > 1600:
                continue
            proteins.append(items[0])
            accessions.append(items[1].split(';')[0])
            maxlen = max(maxlen, len(seq))
            sequences.append(seq)
            grams = list()
            for i in range(len(seq) - gram_len + 1):
                grams.append(vocab[seq[i: (i + gram_len)]])
            index = np.array([AAINDEX[x] for x in seq])
            indexes.append(index)
            ngrams.append(np.array(grams))
            interpros.append(items[3:])
            for item in items[3:]:
                counter[item] += 1
    print('Maximum sequence length: ', maxlen)
    df = pd.DataFrame({
        'proteins': proteins,
        'accessions': accessions,
        'sequences': sequences,
        'ngrams': ngrams,
        'interpros': interpros,
        'indexes': indexes
    })
    print(df)
    df.to_pickle('data/data.pkl')
    dictionary = list()
    for ipro, cnt in counter.items():
        if cnt >= 100:
            dictionary.append(ipro)
    dict_df = pd.DataFrame({'interpros': dictionary})
    print(dict_df)
    dict_df.to_pickle('data/dictionary.pkl')

            
def get_data():
    w = open('data/data.tsv', 'w')
    with gzip.open('data/uniprot_sprot.dat.gz') as f:
        prot_id = ''
        prot_ac = ''
        seq = ''
        annots = list()
        for line in f:
            items = line.decode('utf-8').strip().split('   ')
            if items[0] == 'ID' and len(items) > 1:
                if prot_id != '' and len(annots) > 0:
                    w.write(prot_id + '\t' + prot_ac + '\t' + seq)
                    for ipro_id in annots:
                        w.write('\t' + ipro_id)
                    w.write('\n')
                prot_id = items[1]
                annots = list()
            elif items[0] == 'AC' and len(items) > 1:
                prot_ac = items[1]
            elif items[0] == 'DR' and len(items) > 1:
                items = items[1].split('; ')
                if items[0] == 'InterPro':
                    ipro_id = items[1]
                    annots.append(ipro_id)
            elif items[0] == 'SQ':
                seq = ''
                while True:
                    s = next(f).decode('utf-8').strip().replace(' ', '')
                    if s == '//':
                        break
                    seq += s

        if len(annots) > 0:
            w.write(prot_id + '\t' + prot_ac + '\t' + seq)
            for go_id in annots:
                w.write('\t' + go_id)
            w.write('\n')
        w.close()
    

def get_fasta():
    df = pd.read_pickle('data/data.pkl')

    index = np.arange(len(df))
    np.random.shuffle(index)
    n = 20000
    df = df.iloc[index[:n]]
    with open('data/swissprot.fasta', 'w') as w:
        for row in df.itertuples():
            w.write('>' + row.proteins + '\n')
            w.write(row.sequences + '\n')


if __name__ == '__main__':
    main()
