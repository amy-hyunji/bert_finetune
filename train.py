import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel
from torch.optim import Adam
import torch.nn.functional as F
import os

train_name = "batch_size_16_train_300k_epoch_5_new_data"

train_df = pd.read_csv('./data/n_train.csv', sep=',')
test_df = pd.read_csv('./data/n_test.csv', sep=',')

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

train_df = train_df.sample(frac=1.0, random_state=999) # about 100,000
test_df = test_df.sample(frac=1.0, random_state=999)

class RateDataset(Dataset):
    ''' Amazon Product Dataset '''
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 1]
        label = int(self.df.iloc[idx, 2]-1)
        #print(f"text: {text}, label: {label}")
        return text, label
    
rate_train_dataset = RateDataset(train_df)
print(f"Train dataset: {len(rate_train_dataset)}")
itr_num = len(rate_train_dataset)
train_loader = DataLoader(rate_train_dataset, batch_size=16, shuffle=True, num_workers=2)

device = torch.device("cuda:7")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
#config = BertConfig.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
#model = BertForMultiLabelSequenceClassification(config)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-6)

itr = 1
p_itr = 500
s_itr = 10000
epochs = 5
total_loss = 0
total_len = 0
total_correct = 0

def save_checkpoint(model, save_pth):
    if not os.path.exists(os.path.dirname(save_pth)):
        os.makedirs(os.path.dirname(save_pth))
    torch.save(model.cpu().state_dict(), save_pth)
    model.to(device)

model.train()
for epoch in range(epochs):
    
    for text, label in train_loader:
        optimizer.zero_grad()
        
        # encoding and zero padding
        encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
        for i in range(len(encoded_list)):
           e = encoded_list[i]
           if (len(e) > 512):
               encoded_list[i] = e[:512]
        padded_list =  [e + [0] * (512-len(e)) for e in encoded_list]
        
        sample = torch.tensor(padded_list)
        sample, label = sample.to(device), label.to(device)
#        labels = label.clone().detach()
        labels = torch.tensor(label)
        outputs = model(sample, labels=labels)
        loss, logits = outputs

        pred = torch.argmax(F.softmax(logits), dim=1)
        correct = pred.eq(labels)
        total_correct += correct.sum().item()
        total_len += len(labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        acc = total_correct/total_len    

        if itr % p_itr == 0:
            print("\n############")
            print('[Epoch {}/{}] Iteration {}/{} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, itr, itr_num, total_loss/p_itr, total_correct/total_len))
            print("############\n")
            total_loss = 0
            total_len = 0
            total_correct = 0

        if itr % s_itr == 0:
            # save model
            model_name = "{}_{}_{:.3f}_ckpt.pth".format(epoch, itr, acc)
            print("saving the model.. {}".format(model_name))
            save_checkpoint(model, "./ckpt/{}/{}".format(train_name,model_name))
           
        itr+=1
    model_name = "{}_ckpt.pth".format(epoch)
    print("saving the model.. {}".format(model_name))
    save_checkpoint(model, "./ckpt/{}/{}".format(train_name,model_name))

# evaluation
model.eval()

rate_eval_dataset = RateDataset(test_df)
print(f"Eval dataset: {len(rate_eval_dataset)}")
eval_loader = DataLoader(rate_eval_dataset, batch_size=16, shuffle=False, num_workers=2)

total_loss = 0
total_len = 0
total_correct = 0

for text, label in eval_loader:
    # encoding and zero padding
    encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
    for i in range(len(encoded_list)):
       e = encoded_list[i]
       if (len(e) > 512):
           encoded_list[i] = e[:512]
    padded_list =  [e + [0] * (512-len(e)) for e in encoded_list]
    
    sample = torch.tensor(padded_list)
    sample, label = sample.to(device), label.to(device)
    labels = torch.tensor(label)
    outputs = model(sample, labels=labels)
    _, logits = outputs

    pred = torch.argmax(F.softmax(logits), dim=1)
    correct = pred.eq(labels)
    total_correct += correct.sum().item()
    total_len += len(labels)

print('Test accuracy: ', total_correct / total_len)
