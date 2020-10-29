from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import logging
import re
import gensim
import pickle

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import defaultdict

# Read data from files
train = pd.read_csv("./corpus/imdb/labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)
test = pd.read_csv("./corpus/imdb/testData.tsv", header=0,
                   delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("./corpus/imdb/unlabeledTrainData.tsv", header=0,
                              delimiter="\t", quoting=3)


def review_to_wordlist(p_review, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(p_review, "lxml").get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    #
    # 5. Return a list of words
    return words


def build_data_train_test(data_train, data_test, train_ratio=0.8):
    """
    Loads data and process data into index
    """
    f_revs = []
    f_vocab = defaultdict(float)
    clean_train_reviews = []
    for review_i in data_train["review"]:
        clean_train_reviews.append(review_to_wordlist(review_i,
                                                      remove_stopwords=True))

    clean_test_reviews = []
    for review_i in data_test["review"]:
        clean_test_reviews.append(review_to_wordlist(review_i,
                                                     remove_stopwords=True))

    # Pre-process train data set
    for i in range(len(clean_train_reviews)):
        f_rev = clean_train_reviews[i]
        y = data_train['sentiment'][i]
        orig_rev = ' '.join(f_rev).lower()
        words = set(orig_rev.split())
        for word in words:
            f_vocab[word] += 1
        datum = {'y': y,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': int(np.random.rand() < train_ratio)}
        f_revs.append(datum)

    for i in range(len(clean_test_reviews)):
        f_rev = clean_test_reviews[i]
        orig_rev = ' '.join(f_rev).lower()
        words = set(orig_rev.split())
        for word in words:
            f_vocab[word] += 1
        datum = {'y': -1,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': -1}
        f_revs.append(datum)
    return f_revs, f_vocab


def load_bin_vec(p_model, p_vocab):
    word_vecs = {}
    unk_words = 0

    for word in p_vocab.keys():
        try:
            word_vec = p_model[word]
            word_vecs[word] = word_vec
        except KeyError:
            unk_words = unk_words + 1

    logging.info('unk words: %d' % unk_words)
    return word_vecs


def get_w(word_vecs, k=300):
    vocab_size = len(word_vecs)
    f_word_idx_map = dict()

    f_w = np.zeros(shape=(vocab_size + 2, k), dtype=np.float32)
    f_w[0] = np.zeros((k,))
    f_w[1] = np.random.uniform(-0.25, 0.25, k)

    i = 2
    for word in word_vecs:
        f_w[i] = word_vecs[word]
        f_word_idx_map[word] = i
        i = i + 1
    return f_w, f_word_idx_map


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ''.join(sys.argv))

    revs, vocab = build_data_train_test(train, test)
    max_l = np.max(pd.DataFrame(revs)['num_words'])
    logging.info('data loaded!')
    logging.info('number of sentences: ' + str(len(revs)))
    logging.info('vocab size: ' + str(len(vocab)))
    logging.info('max sentence length: ' + str(max_l))

    # word2vec GoogleNews
    # model_file = os.path.join('vector', 'GoogleNews-vectors-negative300.bin')
    # model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

    # Glove Common Crawl
    model_file = os.path.join('glove.840B.300d.gensim.txt')
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)

    w2v = load_bin_vec(model, vocab)
    logging.info('word embeddings loaded!')
    logging.info('num words in embeddings: ' + str(len(w2v)))

    W, word_idx_map = get_w(w2v, k=model.vector_size)
    logging.info('extracted index from embeddings! ')

    # pickle_file = os.path.join('pickle', 'vader_movie_reviews_glove.pickle3')
    pickle_file = os.path.join('pickle', 'imdb_train_val_test.pickle3')
    pickle.dump([revs, W, word_idx_map, vocab, max_l], open(pickle_file, 'wb'))
    logging.info('dataset created!')
