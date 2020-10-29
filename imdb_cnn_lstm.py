# max len = 56
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd

import imdb_functions as func

from tensorflow import keras


# maxlen = 56
batch_size = 100
nb_epoch = 10
hidden_dim = 120

kernel_size = 3
nb_filter = 60

test = pd.read_csv("./corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'imdb_train_val_test.pickle3')

    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    X_train, X_test, X_dev, y_train, y_dev = func.make_idx_data(revs, word_idx_map, maxlen=maxlen)

    n_train_sample = X_train.shape[0]
    logging.info("n_train_sample [n_train_sample]: %d" % n_train_sample)

    n_test_sample = X_test.shape[0]
    logging.info("n_test_sample [n_train_sample]: %d" % n_test_sample)

    len_sentence = X_train.shape[1]  # 200
    logging.info("len_sentence [len_sentence]: %d" % len_sentence)

    max_features = W.shape[0]
    logging.info("num of word vector [max_features]: %d" % max_features)

    num_features = W.shape[1]  # 400
    logging.info("dimension of word vector [num_features]: %d" % num_features)

    # Keras Model
    # this is the placeholder tensor for the input sequence
    sequence = keras.layers.Input(shape=(maxlen,), dtype='int32')

    # embedded = Embedding(input_dim=max_features, output_dim=num_features,
    # input_length=maxlen, mask_zero=True, weights=[W], trainable=False) (sequence)
    embedded = keras.layers.Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W],
                                      trainable=False)(sequence)
    embedded = keras.layers.Dropout(0.25)(embedded)

    # Convolution
    convolution = keras.layers.Convolution1D(filters=nb_filter,
                                             kernel_size=kernel_size,
                                             padding='valid',
                                             activation='relu',
                                             strides=1
                                             )(embedded)

    max_pooling = keras.layers.MaxPooling1D(pool_size=2)(convolution)

    # LSTM
    lstm = keras.layers.LSTM(hidden_dim, recurrent_dropout=0.25)(max_pooling)

    output = keras.layers.Dense(2, activation='softmax')(lstm)
    model = keras.Model(inputs=sequence, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    model.fit(X_train, y_train, validation_data=[X_dev, y_dev], batch_size=batch_size, epochs=nb_epoch)
    y_pred = model.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": y_pred})

    # Use pandas to write the comma-separated output file
    # result_output.to_csv("./result/bi-lstm.csv", index=False, quoting=3)

    result_output.to_csv("./result/cnn-lstm.csv", index=False, quoting=3)
