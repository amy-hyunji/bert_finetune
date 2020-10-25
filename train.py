from tqdm import tqdm
import torch
import os
from torch import nn
from load_dataset import *
from Models import BertClassifier
#from KoBERT.kobert.pytorch_kobert_adapter import get_pytorch_kobert_model_adapter
from transformers import AdamW
#from transformers.optimization import WarmupLinearSchedule
from transformers.optimization import get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from transformers import AutoTokenizer, TFAutoModel
from pytorch_transformers import BertConfig, BertForSequenceClassification

TENSORBOARD_DIR = "./tensorboard"
if not os.path.exists(TENSORBOARD_DIR):
    os.mkdir(TENSORBOARD_DIR)
task = "rate"
writerDIR = os.path.join(TENSORBOARD_DIR, task)
if not os.path.exists(writerDIR):
    os.mkdir(writerDIR)
writer = SummaryWriter(writerDIR)

if not os.path.exists("./ckpt/{}".format(task)):
    os.makedirs("./ckpt/{}".format(task))

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def prepare_train_adapter(model):
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = True

def save_checkpoint(model, save_pth):
    if not os.path.exists(os.path.dirname(save_pth)):
        os.makedirs(os.path.dirname(save_pth))
    torch.save(model.cpu().state_dict(), save_pth)
    model.cuda()

## Setting parameters
batch_size = 64
warmup_ratio = 0.1
num_epochs = 250
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5
dr_rate = 0.5

device = torch.device("cuda:5")
torch.cuda.set_device(device)

#bertmodel = TFAutoModel.from_pretrained("bert-base-uncased")
#bertmodel, vocab  = get_pytorch_kobert_model_adapter()
#model = BertClassifier.BERTClassifier(bertmodel, dr_rate=dr_rate).to(device)

config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
            num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
num_labels=5
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.to(device)

#prepare_train_adapter(model)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

train_d = load_train()
print("finished loading train")
print(train_d[0])

test_d = load_test()
print(test_d[0])

t_total = len(train_d) * num_epochs
warmup_step = int(t_total * warmup_ratio)
#scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_step, t_total=t_total)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

#sequence_output, pooled_output = model(input_ids, input_mask, token_type_ids)
#pooled_output.shape

print("num of trainable parameters")
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)


for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    steps = len(train_d) // batch_size
    print("total train data : %d" % len(train_d))
    print("total steps %d" % steps)
    #for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm.tqdm(train_d)):
    print("START TRAINING!!!!!!!!!!!!!!!!!!!")
    for batch_id in tqdm(range(steps)):
        batch = train_d[batch_size*batch_id:batch_size*(batch_id+1)]
        token_ids, valid_length, segment_ids, labels  = [], [], [], []

        for i in range(len(batch)):
            print(f"batch[0]: {batch[0][0]}, batch[1]: {batch[1]}")
            token_id = batch[i][0]['input_ids']
            val_len = batch[i][0]['token_type_ids']
            segment_id = batch[i][0]['attention_mask']
            print(f"i: {i}, token_id: {token_id}, val_len: {val_len}, segment_id: {segment_id}")
            token_ids.append(token_id)
            valid_length.append(val_len)
            segment_ids.append(segment_id)

            label = batch[i][1]
            labels.append(label)

        token_ids = torch.LongTensor(token_ids)
        valid_length = torch.LongTensor(valid_length)
        segment_ids = torch.LongTensor(segment_ids)
        labels = torch.LongTensor(labels)

        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length.long().to(device)
        labels = labels.long().to(device)

        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, labels)
        #loss.requires_grad = True
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, labels)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
        # tensorboard
        writer.add_scalar("train/loss", loss, batch_id + steps*e)
    print("epoch {} train acc {}".format(e, train_acc / (batch_id+1)))
    writer.add_scalar("train/accuracy", train_acc/(batch_id+1), e)

    # save model
    if (e%50 == 0):
        model_name = "{}_ckpt.pth".format(e)
        print("saving the model.. {}".format(model_name))
        save_checkpoint(model, "./ckpt/{}".format(model_name))


    model.eval()
    steps = len(test_d) // batch_size
    for batch_id in tqdm(range(steps)):
        batch = test_d[batch_size*batch_id:batch_size*(batch_id+1)]
        token_ids, valid_length, segment_ids, labels  = [], [], [], []

        for el in range(len(batch)):
            token_ids.append(batch[el][0]['input_ids'])
            valid_length.append(batch[el][0]['token_type_ids'])
            segment_ids.append(batch[el][0]['attention_mask'])

            label = batch[el][1]
            labels.append(label)

        token_ids = torch.LongTensor(token_ids).to(device)
        valid_length = torch.LongTensor(valid_length).to(device)
        segment_ids = torch.LongTensor(segment_ids).to(device)
        labels = torch.LongTensor(labels).to(device)

        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, labels)
    print("epoch {} test acc {}".format(e, test_acc / (batch_id+1)))
    writer.add_scalar("test/accuracy", test_acc/(batch_id+1), e)
    writer.close()
    print("done writing")
