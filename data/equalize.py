import os
import random
import pandas as pd
import sys
from tqdm import tqdm

data_path = ["train.csv", "test.csv", "dev.csv"]
data_num = [300000, 30000, 30000]

total = {'text': [], 'label': []}

for split, _num in zip(data_path, data_num):
   label_0 = []
   label_1 = []
   label_2 = []
   label_3 = []
   label_4 = []

   print(f"reading: {split}")
   df = pd.read_csv(split)
   text = df['text']
   label = df['label']

   per_num = int(_num/5)
   print(f"_num: {_num}, per_num: {per_num}")
   for i in tqdm(range(len(text))): 
     if (len(label_0)+len(label_1)+len(label_2)+len(label_3)+len(label_4) == per_num*5):
        break
     if (label[i] == 1.0 and len(label_0) < per_num): 
        try:
           _text = text[i][:512]
           label_0.append([_text, 1]) 
        except:
           print(f"text: {text[i]}")
     elif (label[i] == 2.0 and len(label_1) < per_num): 
        try:
           _text = text[i][:512]
           label_1.append([_text, 2]) 
        except:
           print(f"text: {text[i]}")
     elif (label[i] == 3.0 and len(label_2) < per_num): 
        try:
           _text = text[i][:512]
           label_2.append([_text, 3]) 
        except:
           print(f"text: {text[i]}")
     elif (label[i] == 4.0 and len(label_3) < per_num): 
        try:
           _text = text[i][:512]
           label_3.append([_text, 4]) 
        except:
           print(f"text: {text[i]}")
     elif (label[i] == 5.0 and len(label_4) < per_num): 
        try:
           _text = text[i][:512]
           label_4.append([_text, 5]) 
        except:
           print(f"text: {text[i]}")
     else: 
         continue    

   print("number of elements in...")
   print(f"label_0: {len(label_0)}")
   print(f"label_1: {len(label_1)}")
   print(f"label_2: {len(label_2)}")
   print(f"label_3: {len(label_3)}")
   print(f"label_4: {len(label_4)}")

   _total = []
   _total += label_0
   _total += label_1
   _total += label_2
   _total += label_3
   _total += label_4

   random.shuffle(_total)

   for elem in _total:
      total['text'].append(elem[0])
      total['label'].append(elem[1])

   _df = pd.DataFrame(total)
   print(f"saving ./n_{split}")
   _df.to_csv(f"./n_{split}")
