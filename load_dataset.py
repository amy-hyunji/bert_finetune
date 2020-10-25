import csv
import torch
import numpy as np
import sys
import pandas as pd
from tqdm import tqdm

from pytorch_transformers import BertTokenizer 

max_len = 128 
pad = True
pair = False


def load_train():
    train_dir = './data/train.csv'

    train = []
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    reader = pd.read_csv(train_dir)
    labellist = list(reader['label'])[:100]
    textlist = list(reader['text'])[:100]
    assert (len(textlist) == len(labellist))
    for i in tqdm(range(len(textlist))):
       _text = textlist[i]
       _label = labellist[i]
       newrow = []
       sentence = str(_text) if not isinstance(_text, str) else _text
       if (not isinstance(sentence, str)): 
         print(f"*** not string: type is {type(sentence)}")
         continue
       label = int(float(_label-1)) 
       if (not isinstance(label, int)): 
         print(f"*** not float: {label} but {type(label)}")
         continue
       text = tokenizer(sentence)
       newrow.append(text)
       newrow.append(np.int32(label))
       train.append(newrow)

    return train

def load_test():
    train_dir = './data/test.csv'

    test = []
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    reader = pd.read_csv(train_dir)
    labellist = list(reader['label'])[:10]
    textlist = list(reader['text'])[:10]
    assert (len(textlist) == len(labellist))
    for i in tqdm(range(len(textlist))):
       _text = textlist[i]
       _label = labellist[i]
       newrow = []
       sentence = str(_text) if not isinstance(_text, str) else _text
       if (not isinstance(sentence, str)): 
         print(f"*** not string: type is {type(sentence)}")
         continue
       label = int(float(_label-1)) 
       if (not isinstance(label, int)): 
         print(f"*** not float: {label} but {type(label)}")
         continue
       text = tokenizer(sentence)
       newrow.append(text)
       newrow.append(np.int32(label))
       test.append(newrow)

    return test 


