import numpy as np
from tensorflow import keras


def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(1)
    return x


def make_idx_data(revs, word_idx_map, maxlen=60):
    """
    Transforms sentences into a 2-d matrix.
    """
    x_train, x_test, x_dev, y_train, y_dev = [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        y = rev['y']

        if rev['split'] == 1:
            x_train.append(sent)
            y_train.append(y)
        elif rev['split'] == 0:
            x_dev.append(sent)
            y_dev.append(y)
        elif rev['split'] == -1:
            x_test.append(sent)

    x_train = keras.preprocessing.sequence.pad_sequences(np.array(x_train), maxlen=maxlen)
    x_dev = keras.preprocessing.sequence.pad_sequences(np.array(x_dev), maxlen=maxlen)
    x_test = keras.preprocessing.sequence.pad_sequences(np.array(x_test), maxlen=maxlen)
    # X_valid = sequence.pad_sequences(np.array(X_valid), maxlen=maxlen)
    y_train = keras.utils.to_categorical(np.array(y_train))
    y_dev = keras.utils.to_categorical(np.array(y_dev))
    # y_valid = np.array(y_valid)

    return [x_train, x_test, x_dev, y_train, y_dev]
