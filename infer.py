import numpy as np
import pandas as pd
import torch
import os
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel 
import torch.nn.functional as F

def load_checkpoint(model, ckpt):
   if ckpt is None:
      print("empty checkpoint!")
      sys.exit(-1)
   real_path = os.path.abspath(ckpt)
   model.load_state_dict(torch.load(real_path))

def test(ckpt):
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
   load_checkpoint(model, ckpt)
   model.eval()

   while(1):
      sentence = input("Enter Sentence: ")
      encode = tokenizer.encode(sentence, add_special_tokens=True)
      print(f"encode: {encode}")
      padded = [encode + [0] * (512-len(encode))]
      sentence = torch.tensor(padded)
      label = torch.tensor([0])

      results = model(sentence, label)
      print(f"results: {results[0]}")
      _softmax = F.softmax(results[0], dim=1)
      pred = torch.argmax(F.softmax(results[0], dim=1))
      print(f"The rate is likely to be {pred+1}")

if __name__ == "__main__":
   test("./ckpt/batch_size_8_train_100k_epoch_5/3_280000_0.740_ckpt.pth")
