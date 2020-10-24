import csv
import torch
import numpy as np

from transformers import AutoTokenizer, TFAutoModel

max_len = 128 
pad = True
pair = False


def load_train(vocab):
    train_dir = './data/train.csv'

    train = []
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    with open(train_dir, 'r') as te:
        reader = csv.reader(te, delimiter=',')
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            newrow = []
            sentence = row[1]
            label = int(row[2]) -1
            text = tokenizer(sentence)
            newrow.append(text)
            newrow.append(np.int32(label))
            train.append(newrow)

    return train

def load_test(vocab):
    test_dir = './data/test.csv'

    test = []
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    with open(test_dir, 'r') as te:
        reader = csv.reader(te, delimiter=',')
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            newrow = []
            sentence = row[1]
            label = int(row[2]) -1
            text = tokenizer(sentence)
            newrow.append(text)
            newrow.append(np.int32(label))
            test.append(newrow)

    return test 