#!/usr/bin/env python

"""
python nn_hierarchical_network.py
"""

import numpy as np
import pandas as pd
import click as ck
from keras.models import Sequential, Model, load_model
from keras.layers import (
    Dense, Dropout, Activation, Input,
    Flatten, Highway, BatchNormalization, Reshape)
from keras.layers.embeddings import Embedding
from keras.layers.merge import Dot
from keras.layers.convolutional import (
    Conv1D, MaxPooling1D)
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop, Adadelta
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
from keras import backend as K
import sys
from collections import deque
import time
import logging
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from scipy.spatial import distance
from scipy import sparse
import math
from multiprocessing import Pool

from utils import read_fasta
from aaindex import is_ok, AAINDEX

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
sys.setrecursionlimit(100000)

MAXLEN = 1002

class PairGenerator(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def fit(self, inputs, scores):
        self.start = 0
        self.inputs = inputs
        self.scores = scores
        self.size = len(self.inputs)
        self.index_size = self.size ** 2
        self.index = np.arange(self.index_size)
        np.random.seed(seed=0)
        np.random.shuffle(self.index)
        
    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.size:
            batch_index = np.arange(
                self.start, min(self.index_size, self.start + self.batch_size))
            input1 = np.empty((self.batch_size, MAXLEN, 21), dtype=np.float32)
            input2 = np.empty((self.batch_size, MAXLEN, 21), dtype=np.float32)
            scores = np.empty((self.batch_size, ), dtype=np.float32)
            for i, ind in enumerate(self.index[batch_index]):
                x = ind // self.size
                y = ind % self.size
                input1[i, :] = self.inputs[x, :, :]
                input2[i, :] = self.inputs[y, :, :]
                scores[i] = self.scores[x][y]
                
            return [input1, input2], scores
        else:
            self.reset()
            return self.next()


@ck.command()
@ck.option(
    '--device',
    default='gpu:1',
    help='GPU or CPU device id')
@ck.option('--train', is_flag=True)
def main(device, train):
    global interpros
    df = pd.read_pickle('data/dictionary.pkl')
    interpros = df['interpros'].values
    global nb_classes
    nb_classes = len(interpros)
    global interpro_ix
    interpro_ix = {}
    for i, ipro in enumerate(interpros):
        interpro_ix[ipro] = i 
    # with tf.device('/' + device):
    train_model(is_train=train)


def load_data(split=0.9):

    ngrams = list()
    df = pd.read_pickle('data/sw_scores.pkl')
    prot_index = {}
    for row in df.itertuples():
        seq = row.sequences
        if not is_ok(seq) or len(seq) > MAXLEN:
            continue
        grams = list(map(lambda x: AAINDEX[x], seq))
        ngrams.append(grams)
        prot_index[row.proteins] = len(prot_index)
        
    df['ngrams'] = ngrams
    n = len(df)
    index = np.arange(n)
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_n = int(n * split)
    valid_n = int(train_n * split)
    train_df = df.iloc[index[:valid_n]]
    valid_df = df.iloc[index[valid_n:train_n]]
    test_df = df.iloc[index[train_n:]]
    
    def get_values(df):
        index = np.zeros((len(df),), dtype=np.int32)
        data = np.zeros((len(df), MAXLEN, 21), dtype=np.float32)
        for i, row in enumerate(df.itertuples()):
            for j in range(len(row.ngrams)):
                data[i, j, row.ngrams[j]] = 1
            index[i] = prot_index[row.proteins]
        scores = df['scores'].values
        for i in range(len(scores)):
            scores[i] = scores[i][index]
        return data, scores

    train, valid, test = get_values(train_df), get_values(valid_df), get_values(test_df)
    
    return train, valid, test 


def get_dense_features():
    model = Sequential()
    model.add(Reshape((MAXLEN * 21, ), input_shape=(MAXLEN, 21)))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    return model
    

def get_feature_model():
    embedding_dims = 128
    max_features = 8001
    model = Sequential()
    model.add(Embedding(
        max_features,
        embedding_dims,
        input_length=MAXLEN))
    model.add(Conv1D(
        filters=32,
        kernel_size=128,
        padding='valid',
        activation='relu',
        strides=1))
    model.add(MaxPooling1D(pool_size=64, strides=32))
    model.add(Flatten())
    return model

def merge_outputs(outputs, name):
    if len(outputs) == 1:
        return outputs[0]
    return merge(outputs, mode='concat', name=name, concat_axis=1)


def merge_nets(nets, name):
    if len(nets) == 1:
        return nets[0]
    return merge(nets, mode='sum', name=name)


def get_function_node(name, inputs):
    output_name = name + '_out'
    # net = Dense(256, name=name, activation='relu')(inputs)
    output = Dense(1, name=output_name, activation='sigmoid')(inputs)
    return output, output



def get_model():
    logging.info("Building the model")
    input1 = Input(shape=(MAXLEN, 21), dtype='float32', name='input1')
    input2 = Input(shape=(MAXLEN, 21), dtype='float32', name='input2')
    feature_model = get_dense_features()
    vector1 = feature_model(input1)
    vector2 = feature_model(input2)
    net = Dot(axes=1)([vector1, vector2])
    net = Activation('sigmoid')(net)
    
    model = Model(inputs=[input1, input2], outputs=net)
    model.summary()
    logging.info('Compiling the model')
    optimizer = RMSprop()

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy')
    logging.info(
        'Compilation finished')
    return model


def train_model(batch_size=256, epochs=100, is_train=True):
    # set parameters:
    start_time = time.time()
    logging.info("Loading Data")
    train, valid, test = load_data()
    train_data, train_scores = train
    valid_data, valid_scores = valid
    test_data, test_scores = test

    logging.info("Data loaded in %d sec" % (time.time() - start_time))
    logging.info("Training data size: %d" % train_data.shape[0])
    logging.info("Validation data size: %d" % valid_data.shape[0])
    logging.info("Test data size: %d" % test_data.shape[0])

    model_path = 'data/model_cosine.h5'
    checkpointer = ModelCheckpoint(
        filepath=model_path,
        verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    logging.info('Starting training the model')

    train_generator = PairGenerator(batch_size)
    train_generator.fit(train_data, train_scores)
    valid_generator = PairGenerator(batch_size)
    valid_generator.fit(valid_data, valid_scores)
    test_generator = PairGenerator(batch_size)
    test_generator.fit(test_data, test_scores)
    
    if is_train:
        valid_steps = int(math.ceil(valid_data.shape[0] ** 2 / batch_size))
        train_steps = int(math.ceil(train_data.shape[0] ** 2 / batch_size))
        model = get_model()
        model.fit_generator(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=valid_generator,
            validation_steps=valid_steps,
            max_queue_size=batch_size,
            workers=12,
            callbacks=[checkpointer, earlystopper])

    logging.info('Loading best model')
    model = load_model(model_path)

    logging.info('Testing')
    test_steps = int(math.ceil(test_data.shape[0] ** 2 / batch_size))
    loss = model.evaluate_generator(
        test_generator, steps=test_steps, verbose=1)

    logging.info('Test loss:', loss)
    
def compute_roc(preds, labels):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_mcc(preds, labels):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc


def compute_performance(preds, labels):
    preds = np.round(preds, 2)
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        total = 0
        f = 0.0
        p = 0.0
        r = 0.0
        p_total = 0
        for i in range(labels.shape[0]):
            tp = np.sum(predictions[i, :] * labels[i, :])
            fp = np.sum(predictions[i, :]) - tp
            fn = np.sum(labels[i, :]) - tp
            # all_gos = set()
            # for go_id in gos[i]:
            #     if go_id in all_functions:
            #         all_gos |= get_anchestors(go, go_id)
            # all_gos.discard(GO_ID)
            # all_gos -= func_set
            # fn += len(all_gos)
            if tp == 0 and fp == 0 and fn == 0:
                continue
            total += 1
            if tp != 0:
                p_total += 1
                precision = tp / (1.0 * (tp + fp))
                recall = tp / (1.0 * (tp + fn))
                p += precision
                r += recall
        if p_total == 0:
            continue
        r /= total
        p /= p_total
        if p + r > 0:
            f = 2 * p * r / (p + r)
            if f_max < f:
                f_max = f
                p_max = p
                r_max = r
                t_max = threshold
                predictions_max = predictions
    return f_max, p_max, r_max, t_max, predictions_max





if __name__ == '__main__':
    main()
