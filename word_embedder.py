from elmoformanylangs import Embedder
from conllu import parse_incr
import os
import re
import numpy as np
import pandas as pd
from io import open
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json


def data_extractor(tokens):
    word_data = []
    for token in tokens:
        temp_word_data = []

        for i in token:
            temp_word_data.append(i['form'])

        word_data.append(temp_word_data)
    return word_data


def word_tokenizer(data):
    word_tokenizer = Tokenizer(oov_token="<OOV>")
    word_tokenizer.fit_on_texts(data)
    word_index = word_tokenizer.word_index
    train_word_sequences = word_tokenizer.texts_to_sequences(data)
    padded_word_sequences_dis = pad_sequences(
        train_word_sequences, padding='post', maxlen=52, truncating="post")
    return word_index, padded_word_sequences_dis


def savecsv(language, embedding_vectors):
    df = pd.DataFrame(embedding_vectors)

    # saving the dataframe
    df.to_csv('embedding/'+language+'.csv')


def data_preprocesser(filename):
    tokenlist = []
    test_tokenlist = []

    disease_f = os.listdir('/home/ug2018/cse/18075072/ud_data/ud-treebanks-v2.8/'+filename)
    dir_f = []
    for i in disease_f:
        if re.search(".conllu$", i):
            dir_f.append(i)
            print(i)
    files = [filename+"/" + x for x in dir_f]

    for file_link in files:
        data_file = open(file_link, "r", encoding="utf-8")
        if re.search("dev.conllu$", file_link):
            for token in parse_incr(data_file):
                test_tokenlist.append(token)

        else:
            for token in parse_incr(data_file):
                tokenlist.append(token)
    tokenlist = tokenlist + test_tokenlist
    return tokenlist


if __name__ == "__main__":

    filename = 'UD_English-EWT'
    language = "English-EWT"
    tokenlist = data_preprocesser(filename)
    word_data = data_extractor(tokenlist)
    word_index, sentence = word_tokenizer(word_data)

    k = Embedder(
        '144/', batch_size=32)

    embedding_vectors = {}
    # word_index = {'the': 0, 'are': 1, 'is': 2}
    test_word = [[]]

    for word, index in word_index.items():
        test_word[0].append(word)

    diction = k.sents2elmo(test_word)

    for word, index in word_index.items():
        embedding_vectors[word] = list(diction[0][index])
        temp = np.asarray(embedding_vectors[word])

    savecsv(language, embedding_vectors)
