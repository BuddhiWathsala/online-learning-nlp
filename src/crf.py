from collections import Counter
from keras.models import Sequential

from keras.layers import Embedding
from keras_contrib.layers import CRF
from keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
import numpy as np
import time
import sys

ROOT = "../resources/"
EMBEDDING_DIMENSION = 512
MODEL_NAME = "crf_model.hd5"
RESULT_FILE = "CRF_RESULTS.txt"
ENCODING = 'utf-16'
DATA_FILE = 'train1.txt'
TEST_FILE = 'test1.txt'
VOCABULARY_FILE = 'vocabulary.txt'
LABEL_FILE = 'labels.txt'
MAX_SENTENCE_LEN = 150
NO_OF_TOKENS = 3
VOCABULARY_SIZE = 15000


def file_read(file_name):
    with open(file_name, encoding=ENCODING, mode="r+") as file:
        file_text = file.read()
    return file_text


def get_ner_data_and_tokens(initial, data_file):
    text = []
    tokens = []
    tokens_list = []
    max_length = 0
    min_freq = 2
    word_set = []
    no_of_s = 0
    file_text = file_read(data_file)
    data = [[row.split() for row in sample.split('\n')] for sample in file_text.strip().split('\n\n')]
    for sentence in file_text.strip().split('\n\n'):
        no_of_s += 1
        for row in sentence.strip().split('\n'):
            try:
                row_tokens = [w.strip() for w in row.strip().split()]
                if row_tokens[-1] not in tokens_list:
                    tokens_list.append(row_tokens[-1])
            except IndexError:
                pass
    print(tokens_list)
    print(no_of_s)
    if not initial:
        previous_vocabulary = file_read(ROOT + VOCABULARY_FILE)
        for row in previous_vocabulary.strip().split('\n'):
            try:
                row_tokens = [w.strip() for w in row.strip().split()]
                word_set.append(row_tokens[0])
            except IndexError:
                pass

    vocab_file = open(ROOT + VOCABULARY_FILE, "w+", encoding=ENCODING)

    for sentence in file_text.strip().split('\n\n'):
        sentence_text = []
        sentence_tokens = []
        for row in sentence.strip().split('\n'):
            try:
                row_tokens = [w.strip() for w in row.strip().split()]
                sentence_text.append(row_tokens[0])
                word_set.append(row_tokens[0])
                sentence_tokens.append([tokens_list.index(row_tokens[-1])])

            except IndexError:
                pass
        if max_length < len(sentence_text):
            max_length = len(sentence_text)
            print(sentence_text)
        text.append(sentence_text)
        tokens.append(sentence_tokens)
    word_counts = Counter(word_set)
    vocab = ['<pad>', '<unk>'] + [w for w, f in iter(word_counts.items()) if f >= min_freq]
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = []
    for s in data:
        lst = []
        for w in s:
            try:
                lst.append(word2idx.get(w[0], 1))
            except IndexError:
                lst.append(1)
        x.append(lst)
    for v in word_set:
        vocab_file.write(v + "\n")
    vocab_file.close()

    if initial:
        label_file = open(ROOT + LABEL_FILE, "w+", encoding=ENCODING)
        for v in tokens_list:
            label_file.write(v + "\n")
        label_file.close()
    print(max_length)
    return [x, tokens, max_length, len(tokens_list), vocab]


def get_test_map(test_file):
    vocabulary_file = file_read(ROOT + VOCABULARY_FILE)
    file_text = file_read(test_file)
    data = [[row.split() for row in sample.split('\n')] for sample in file_text.strip().split('\n\n')]
    word_set = []
    min_freq = 2
    for row in vocabulary_file.strip().split('\n'):
        try:
            row_tokens = [w.strip() for w in row.strip().split()]
            word_set.append(row_tokens[0])
        except IndexError:
            pass
    word_counts = Counter(word_set)
    vocab = ['<pad>', '<unk>'] + [w for w, f in iter(word_counts.items()) if f >= min_freq]
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = []
    for s in data:
        lst = []
        for w in s:
            try:
                lst.append(word2idx.get(w[0], 1))
            except IndexError:
                lst.append(1)
        x.append(lst)
    return x


def train_model(no_of_epochs, tt_factor, tv_factor, initial, data_file):
    print("start..")
    data = get_ner_data_and_tokens(initial, data_file)
    vectors = data[0]
    tokens = data[1]
    total_data_lines = len(vectors)

    train_test_factor = int(total_data_lines - total_data_lines * tt_factor)
    train_validation_factor = int(train_test_factor - train_test_factor * tv_factor)

    test_tokens_before_padding = tokens[train_test_factor:]
    print(test_tokens_before_padding)
    vectors = pad_sequences(vectors, MAX_SENTENCE_LEN, padding='post')
    tokens = pad_sequences(data[1], MAX_SENTENCE_LEN, value=-1, padding='post')

    train_vectors = vectors[:train_test_factor]
    train_tokens = tokens[:train_test_factor]
    test_vectors = vectors[train_test_factor:]
    test_tokens = tokens[train_test_factor:]
    validation_vectors = vectors[train_validation_factor:train_test_factor]
    validation_tokens = tokens[train_validation_factor:train_test_factor]
    print(test_tokens)
    if initial:
        model = Sequential()
        model.add(Embedding(VOCABULARY_SIZE, EMBEDDING_DIMENSION, mask_zero=True))
        crf = CRF(NO_OF_TOKENS, sparse_target=True)
        model.add(crf)
        model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
        file = open(ROOT + RESULT_FILE, "w+")
    else:

        model = Sequential()
        model.add(Embedding(VOCABULARY_SIZE, EMBEDDING_DIMENSION, mask_zero=True))
        crf = CRF(MAX_SENTENCE_LEN, sparse_target=True)
        model.add(crf)
        model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])

        model.load_weights(ROOT + MODEL_NAME)
        file = open(ROOT + RESULT_FILE, "a+")

    time_lst = []
    start_time = time.time()
    model.fit(
        train_vectors,
        train_tokens,
        epochs=no_of_epochs,
        validation_data=(validation_vectors, validation_tokens)
    )
    time_lst.append(time.time() - start_time)
    predict = model.predict(test_vectors)
    for i in range(len(test_tokens_before_padding)):
        for j in range(len(test_tokens_before_padding[i])):
            if test_tokens[i][j][0] == -1:
                num = 0
            else:
                num = test_tokens[i][j][0]
            file.write(str(predict[i][j].argmax()) + "\t" + str(num) + "\n")
        file.write("\n")
    file.write("##########\n")
    file.close()
    time_file = open(ROOT + "CRF_TIME.txt", "w+")
    for t in time_lst:
        time_file.write(str(t) + " ")
    time_file.close()
    model.save_weights(ROOT + MODEL_NAME, overwrite=True)


def plot_graph(name, lst):
    x = np.arange(1, int(len(lst)+1), 1)
    plt.plot(x, lst, 'b--', x, lst, 'bs')
    plt.ylim(0, 100)
    plt.xlim(0, int(len(lst)+1), 1)
    plt.xlabel("Mini Batch")
    plt.ylabel(name + " %")
    plt.title(name + " Value")
    plt.show()


def accuracies(data_file):
    total_instances = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    precision_arr = []
    recall_arr = []
    F_arr = []
    with open(data_file, encoding="utf-8", mode="r+") as file:
        for read_line in file:
            if read_line != '\n':
                if read_line.strip() == "##########":
                    precision = (TP * 100 / (TP + FP)) if (TP + FP) != 0 else 0.0
                    recall = (TP * 100 / (TP + FN)) if (TP + FN) != 0 else 0.0
                    beta = 1
                    if precision == 0 and recall == 0:
                        F = 0.0
                    else:
                        F = (1 + pow(beta, 2)) * (precision * recall) / (pow(beta, 2) * recall + precision)
                    precision_arr.append(round(precision, 4))
                    recall_arr.append(round(recall, 4))
                    F_arr.append(round(F, 4))
                    TP = 0
                    TN = 0
                    FP = 0
                    FN = 0
                    total_instances = 0
                    continue
                numbers = [float(w) for w in read_line.split()]
                total_instances += 1
                if numbers[1] == 0:
                    if numbers[0] == numbers[1]:
                        TN += 1
                    else:
                        FN += 1
                else:
                    if numbers[0] == numbers[1]:
                        TP += 1
                    else:
                        FP += 1
    return [precision_arr, recall_arr, F_arr]


def report():
    accuracy_arr = accuracies(ROOT + RESULT_FILE)
    # plot_graph("Precision", accuracy_arr[0])
    # plot_graph("Recall", accuracy_arr[1])
    plot_graph("F-Measure", accuracy_arr[2])


def predict_model(test_file):
    test_vectors = get_test_map(test_file)
    test_vectors = pad_sequences(test_vectors, MAX_SENTENCE_LEN)
    model = Sequential()
    model.add(Embedding(VOCABULARY_SIZE, EMBEDDING_DIMENSION, mask_zero=True))
    crf = CRF(MAX_SENTENCE_LEN, sparse_target=True)
    model.add(crf)
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    model.load_weights(ROOT + MODEL_NAME)
    predict = model.predict(test_vectors)

    test_text = file_read(test_file)
    test_f = open(ROOT + TEST_FILE, "w+", encoding=ENCODING)
    label_file = file_read(ROOT+LABEL_FILE)
    labels = []
    test_list = test_text.strip().split('\n\n')
    for l in label_file.split('\n'):
        labels.append(l)
    print(len(test_list[1].split("\n")))
    print(len(predict[1]))
    for i in range(len(test_list)):
        sentence = test_list[i].split('\n')
        for j in range(len(sentence)):
            try:
                row_tokens = [w.strip() for w in sentence[j].split()]
                row_tokens.append(str(labels[int(predict[i][j].argmax())]))

                for r in row_tokens:
                    test_f.write(str(r) + " ")
                test_f.write("\n")

            except IndexError as e:
                return

        test_f.write("\n\n")

    test_f.close()


def main(train, data_file, epochs=1, tt_factor=0.1, tv_factor=0.1, initial=True):
    print(train, data_file, epochs, tt_factor, tv_factor, initial)
    if train:
        train_model(epochs, tt_factor, tv_factor, initial=initial, data_file=data_file)
        report()
    else:
        predict_model(data_file)


if __name__ == "__main__":
    print(sys.argv)
    T = ["True", "1", "T", "t", "true"]
    F = ["False", "0", "F", "f", "false"]
    if sys.argv[1] in T:
        if len(sys.argv) > 3:
            main(
                True,
                sys.argv[2],
                epochs=int(sys.argv[3]),
                tt_factor=float(sys.argv[4]),
                tv_factor=float(sys.argv[5]),
                initial=(True if sys.argv[6] in T else False)

            )
        else:
            main(True, sys.argv[2])
    elif sys.argv[1] in F:
        if len(sys.argv) > 3:
            main(
                False,
                sys.argv[2],
                epochs=int(sys.argv[3]),
                tt_factor=float(sys.argv[4]),
                tv_factor=float(sys.argv[5]),
                initial=(True if sys.argv[6] in T else False)

            )
        else:
            main(False, sys.argv[2])

