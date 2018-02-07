#!/usr/bin/env python

from __future__ import print_function
import argparse
import pandas as pd
import numpy as np
import click as ck
from aaindex import MAXLEN
import sys
import math
from utils import DataGenerator
from scipy import sparse

from keras.layers import (
    Conv1D, MaxPooling1D, UpSampling1D, Reshape, Input,
    Dense, Embedding, Masking, LSTM, RepeatVector)
from keras.models import load_model, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

@ck.command()
@ck.option(
    '--batch-size',
    help='Batch size for training and testing', default=128)
@ck.option(
    '--epochs',
    help='Number of epochs for training', default=12)
@ck.option(
    '--model-file',
    help='Batch size for training and testing', default='model.h5')
@ck.option(
    '--is-train', help="Training mode", is_flag=True, default=False)
@ck.option(
    '--data-file', help="Pandas pickled data file", default='data/data.pkl')
def main(batch_size, epochs, model_file, is_train, data_file):
    train_data, test_data = load_data(data_file)
    if is_train:
        train(train_data, batch_size, epochs, model_file)

    test(test_data, batch_size, model_file)


def load_data(data_file, split=0.8):
    df = pd.read_pickle(data_file)
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

    def get_values(df):
        n = len(df)
        rows = []
        cols = []
        data = []
        for i, row in enumerate(df.itertuples()):
            for ind, j in enumerate(row.indexes):
                rows.append(i)
                cols.append(ind)
                data.append(j / 20.0)
        
        data = sparse.csr_matrix((data, (rows, cols)), shape=(n, MAXLEN))
        index = np.arange(n)
        np.random.seed(seed=0)
        np.random.shuffle(index)
        print(data[0, :].toarray())
        return data[index, :]

    def get_values_one_hot(df):
        n = len(df)
        rows = []
        cols = []
        data = []
        for i, row in enumerate(df.itertuples()):
            for ind, j in enumerate(row.indexes):
                rows.append(i)
                
                cols.append(ind * 21 + j)
                data.append(1)
            for ind in range(len(row.indexes), MAXLEN):
                rows.append(i)
                cols.append(ind * 21)
                data.append(1)
        
        data = sparse.csr_matrix((data, (rows, cols)), shape=(n, MAXLEN * 21))
        index = np.arange(n)
        np.random.seed(seed=0)
        np.random.shuffle(index)
        print(data[0, :].toarray().reshape(MAXLEN, 21))
        return data[index, :]

    train_data = get_values(train_df)
    test_data = get_values(test_df)
    
    return train_data, test_data


def build_model():
    input_seq = Input(shape=(MAXLEN * 21,))
    x = Reshape((MAXLEN, 21))(input_seq)
    x = Conv1D(320, 7, activation='relu', padding='same')(x)
    print(x.get_shape())
    x = MaxPooling1D(4, padding='same')(x)
    print(x.get_shape())
    x = Conv1D(160, 7, activation='relu', padding='same')(x)
    print(x.get_shape())
    x = Conv1D(320, 7, activation='relu', padding='same')(x)
    print(x.get_shape())
    x = UpSampling1D(4)(x)
    print(x.get_shape())
    x = Conv1D(21, 7, activation='sigmoid', padding='same')(x)
    print(x.get_shape())
    
    decoded = Reshape((MAXLEN * 21,))(x)
    
    autoencoder = Model(input_seq, decoded)
    autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')
    autoencoder.summary()
    return autoencoder


def build_model_lstm():
    input_seq = Input(shape=(MAXLEN,))
    x = Reshape((MAXLEN, 1))(input_seq)
    x = Masking(mask_value=0)(x)
    x = LSTM(256)(x)
    x = RepeatVector(MAXLEN)(x)
    x = LSTM(1, return_sequences=True)(x)
    
    decoded = Reshape((MAXLEN,))(x)
    autoencoder = Model(input_seq, decoded)
    autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')
    autoencoder.summary()
    return autoencoder


def build_model_dense():
    input_seq = Input(shape=(MAXLEN * 21,))
    x = Dense(8000, activation='relu')(input_seq)
    x = Dense(4000, activation='relu')(x)
    x = Dense(8000, activation='relu')(x)
    decoded = Dense(MAXLEN * 21, activation='relu')(x)
    autoencoder = Model(input_seq, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.summary()
    return autoencoder


def train(data, batch_size, epochs, model_file, validation_split=0.8):
    index = np.arange(data.shape[0])
    train_n = int(data.shape[0] * validation_split)
    train_data, valid_data = data[index[:train_n], :], data[index[train_n:], :]
    train_generator = DataGenerator(batch_size)
    train_generator.fit(train_data, train_data)
    valid_generator = DataGenerator(batch_size)
    valid_generator.fit(valid_data, valid_data)
    steps = int(math.ceil(train_n / batch_size))
    valid_n = data.shape[0] - train_n
    valid_steps = int(math.ceil(valid_n / batch_size))

    checkpointer = ModelCheckpoint(
        filepath=model_file,
        verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=100, verbose=1)

    model = build_model_lstm()
    model.fit_generator(
        train_generator,
        steps_per_epoch=steps, epochs=epochs,
        validation_data=valid_generator, validation_steps=valid_steps,
        callbacks=[earlystopper, checkpointer])
    

def test(data, batch_size, model_file):
    model = load_model(model_file)
    generator = DataGenerator(batch_size)
    generator.fit(data, data)
    steps = int(math.ceil(data.shape[0] / batch_size))
    loss = model.evaluate_generator(generator , steps=steps)
    print('Test loss %f' % (loss, ))

    preds = model.predict_generator(generator, steps=steps)
    # preds = preds.reshape(data.shape[0], MAXLEN, 21)
    # preds = np.argmax(preds, axis=2)
    real = data.toarray().reshape(data.shape[0], MAXLEN)
    # real = np.argmax(real, axis=2)
    preds = (preds * 20).astype(np.int32)
    real = (real * 20).astype(np.int32)
    for i in range(10):
        c = 0
        l = 0
        lp = 0
        for j in range(len(real[i])):
            if real[i, j] != 0 and real[i, j] == preds[i, j]:
                c += 1
            elif l == 0 and real[i, j] == 0:
                l = j
            if lp == 0 and preds[i, j] == 0:
                lp = j
        print('Match %d, Length %d, Length pred %d' % (c, l, lp))
    


if __name__ == '__main__':
    main()
