import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel
from torch.optim import Adam
import torch.nn.functional as F
import os

"""
class BertForMultiLabelSequenceClassification(PreTrainedBertModel):

    def __init__(self, config, num_labels=5):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits
        
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True
"""



train_name = "batch_size_8_train_100k_epoch_5"

train_df = pd.read_csv('./data/train.csv', sep=',')
test_df = pd.read_csv('./data/test.csv', sep=',')

"""
train_df = pd.read_csv('./nsmc/ratings_train.txt', sep='\t')
test_df = pd.read_csv('./nsmc/ratings_test.txt', sep='\t')
"""

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

#train_df = train_df.sample(frac=0.1, random_state=999)
#test_df = test_df.sample(frac=0.1, random_state=999)

train_df = train_df.sample(frac=0.031, random_state=999) # about 100,000
test_df = test_df.sample(frac=0.031, random_state=999)

class NsmcDataset(Dataset):
    ''' Naver Sentiment Movie Corpus Dataset '''
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 1]
        label = int(self.df.iloc[idx, 2]-1)
        #print(f"text: {text}, label: {label}")
        return text, label
    
nsmc_train_dataset = NsmcDataset(train_df)
print(f"Train dataset: {len(nsmc_train_dataset)}")
itr_num = len(nsmc_train_dataset)
train_loader = DataLoader(nsmc_train_dataset, batch_size=8, shuffle=True, num_workers=2)

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
        
        if itr % p_itr == 0:
            print("\n############")
            print('[Epoch {}/{}] Iteration {}/{} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, itr, itr_num, total_loss/p_itr, total_correct/total_len))
            print("############\n")
            total_loss = 0
            total_len = 0
            total_correct = 0

        if itr % s_itr == 0:
            # save model
            model_name = "{}_{}_ckpt.pth".format(epoch, itr)
            print("saving the model.. {}".format(model_name))
            save_checkpoint(model, "./ckpt/{}/{}".format(train_name,model_name))
           
        itr+=1
    model_name = "{}_ckpt.pth".format(epoch)
    print("saving the model.. {}".format(model_name))
    save_checkpoint(model, "./ckpt/{}/{}".format(train_name,model_name))
# evaluation
model.eval()

nsmc_eval_dataset = NsmcDataset(test_df)
print(f"Eval dataset: {len(nsmc_eval_dataset)}")
eval_loader = DataLoader(nsmc_eval_dataset, batch_size=8, shuffle=False, num_workers=2)

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
