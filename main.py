from Bio import SeqIO
import torch
import time
import random
import numpy as np
from propy.PseudoAAC import GetAPseudoAAC
from propy.AAComposition import CalculateDipeptideComposition
import torch.nn.functional as F
from torch import nn
from torch import tensor
from propy.PseudoAAC import GetPseudoAAC
from catboost import CatBoostRegressor, Pool, CatBoostClassifier, cv
from sklearn.model_selection import train_test_split
import sklearn
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModel
import datetime

device=torch.device("cuda:2")
aa = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]


def read_data(path):
    records = list(SeqIO.parse(path,format="fasta"))
    aa_seq = [str(seq_dic.seq) for seq_dic in records]
    return aa_seq


def get_DPC(seq):
    DPC= []
    dipeptides = [seq[i] + seq[i+1] for i in range(len(seq)-1)]
    for i in aa:
        for j in aa:
            dipeptide = i + j
            DPC.append(round(dipeptides.count(dipeptide)/(len(seq)-1) * 100, 2))
    DPC = tensor(DPC)
    return DPC


def get_BPF(seq):
    seq_len = min(7, len(seq))
    seq_idx = [aa.index(c) for c in seq]
    BPF = F.one_hot((tensor(seq_idx[:seq_len])).to(torch.int64), num_classes=20)
    BPF = BPF.view(-1)
    BPF = F.pad(BPF, (0, 140-len(BPF)))
    BPF = BPF.to(torch.float)
    return BPF


def get_PAAC(seq):
    seq_len = min(7, len(seq))
    # AAP=[_Hydrophobicity, _hydrophilicity, _residuemass, _pK1, _pK2, _pI]
    PAAC = GetPseudoAAC(seq, lamda=seq_len)
    PAAC = F.pad(tensor(list(PAAC.values())), (0, 27 - len(PAAC)))
    return PAAC


def get_APAAC(seq):
    seq_len = min(7, len(seq))
    PAAC = GetAPseudoAAC(seq, lamda=seq_len)
    PAAC = F.pad(tensor(list(PAAC.values())), (0, 34 - len(PAAC)))
    return PAAC


def get_KMER(seq):
    kmer = []
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dic = {c:str(idx) for idx, group in enumerate(groups) for c in group}
    seq = ''.join([group_dic[c] for c in seq])
    seq_kmers = [seq[i] + seq[i+1] + seq[i+2] for i in range(len(seq)-2)]
    all_kmers = [c1 + c2 + c3 for c1 in '0123456' for c2 in '0123456' for c3 in '0123456']
    kmer = [[1 if all_kmer == seq_kmer else 0 for all_kmer in all_kmers] for seq_kmer in seq_kmers]
    kmer = tensor(kmer, dtype=torch.float)
    u, s, v = torch.linalg.svd(kmer)
    kmer = torch.sum(kmer, 0) / len(seq_kmers)
    return kmer


def get_feature_1(seq):
    dpc = get_DPC(seq)
    bpf = get_BPF(seq)
    paac = get_PAAC(seq)
    apaac = get_APAAC(seq)
    kmer = get_KMER(seq)
    input_boost = torch.cat((dpc, bpf, paac, apaac, kmer))
    return input_boost.numpy()


def get_feature_2(seq):
    seq = [aa.index(c)+1 for c in seq]
    seq = seq[:50] if len(seq) > 50 else seq + [0] * (50-len(seq))
    return np.array(seq)


aug_dic = {'A':'V', 'S':'T', 'F':'Y', 'K':'R', 'C':'M', 'D':'E', 'N':'Q', 'V':'I'}
# aug_dic_2 = {v: k for k, v in aug_dic.items()}
# aug_dic.update(aug_dic_2)
# aug_dic['A'] = 'V'



def aug(seqs):
    aug_seqs = []
    for seq in seqs:
        aug_seq = []
        nums = np.random.choice([0, 1], size=len(seq), p=[0.5, 0.5])
        for i, c in enumerate(seq):
            c = aug_dic[c] if nums[i] and c in aug_dic else c
            aug_seq.append(c)
        aug_seqs.append(''.join(aug_seq))
    return aug_seqs


def aug_a(seqs):
    aug_seqs = []
    for seq in seqs:
        aug_seq = []
        nums = np.random.choice([0, 1], size=len(seq), p=[0.5, 0.5])
        for i, c in enumerate(seq):
            c = 'A' if nums[i] else c
            aug_seq.append(c)
        aug_seqs.append(''.join(aug_seq))
    return aug_seqs

class acplstm(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(acplstm, self).__init__()
        # self.bert = AutoModel.from_pretrained("preberts/chenberta_base")
        self.bert = AutoModel.from_pretrained("preberts/smiles_tokenized")
        # self.bert = AutoModel.from_pretrained("preberts/smiles_BPE")
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(21, embedding_dim) 
        self.num_layers = 3 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
        # self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.num_layers)
        # self.bn1 = nn.BatchNorm1d(944)
        # self.bn2 = nn.BatchNorm1d(self.hidden_dim * 2 + 944)
        # self.dropout
        self.linear_1 = nn.Linear(self.hidden_dim * 2 + 944 + 768 + 50, 1024)
        self.linear_2 = nn.Linear(1024, 512)
        self.linear_3 = nn.Linear(512, 256)
        self.linear_4 = nn.Linear(256, 2)
        # self.linear3 = nn.Linear()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)

        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, x, hidden=None):
        x1 = x[:, 944:994].to(torch.long)
        x2 = x[:, 994:].to(torch.int64)
        x3 = x[:, :944]
        embeds = self.embedding(x1)
        x2 = self.bert(x2)
        x2 = x2.pooler_output
        # x2 = F.relu()
        # print(x1.shape) torch.Size([b, 50])
        # print(embeds.shape) torch.Size([b, 50, 512])
        # self.bert(x2).logits.shape   torch.Size([b, 512, 767])
        lstm_out, (h_n, c_n) = self.lstm(embeds, hidden)
        # print(lstm_out.shape, h_n.shape, c_n.shape) ([b, 50, 1024]) ([6, b, 512]) [6, b, 512])
        h_n_l = h_n[-2]
        h_n_r = h_n[-1]
        x4 = torch.cat([h_n_l, h_n_r], dim=-1)
        # x4 = self.dropout2(x4)
        # output = self.bn1(h_n_lr)
        # h_n_l torch.Size([b, 512])
        # h_n_r torch.Size([b, 512])
        # h_n_lr torch.Size([b, 1024])
        # output = self.dropout(h_n_lr)
        output = torch.cat([x1, x2, x3, x4], dim=-1)
        # output = self.dropout1(output)
        # output = self.bn2(output)
        output = F.relu(self.linear_1(output))
        output = F.relu(self.linear_2(output))
        output = F.relu(self.linear_3(output))
        output = self.linear_4(output)
        # output = F.softmax(output, dim=1)
        # output = self.relu(self.linear2(output))
        # output = self.linear3(output)
        return output, (h_n.data, c_n.data)


def train(dataloader, model, loss_fn, optimizer, hidden):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred, hidden = model(X, hidden)
        hidden = None
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return hidden


def test(dataloader, model, hidden):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred, _ = model(X, hidden)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).to(torch.float).sum().item()
    test_loss /= size
    correct /= size
#     if correct >= 0.90:
#         torch.save(model, './model'+str(datetime.datetime.now().strftime('%H%M%S')) + str(correct))
#     print(f"correct = {correct}, Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


acp250_path = "./data/acp240.txt"
nacp250_path = "./data/nacp240.txt"
# acp_seqs = read_data(acp250_path)
# nacp_seqs = read_data(nacp250_path)
acp740_path = "./data/acp740.txt"
nacp740_path = "./data/non-acp740.txt"
acp_seqs = read_data(acp740_path)
nacp_seqs = read_data(nacp740_path)
train_data = acp_seqs + nacp_seqs
train_label = [1] * len(acp_seqs) + [0] * len(nacp_seqs)
print('split...')
X_train, X_test, y_train, y_test =train_test_split(train_data, train_label, test_size=0.2, random_state=5)
y_train = y_train + y_train + y_train
X_train = X_train + aug(X_train) + aug_a(X_train)

if os.path.exists("acp740trainf1.npy") and os.path.exists("acp740trainf2.npy"):
    print('loading')
    train_data_1 = np.load("acp740trainf1.npy")
    train_data_2 = np.load("acp740trainf2.npy")
    test_data_1 = np.load("acp740testf1.npy")
    test_data_2 = np.load("acp740testf2.npy")
else:
    print('processing...')
    print('acp740f1...')
    train_data_1 = [get_feature_1(seq) for seq in X_train]
    train_data_2 = [get_feature_2(seq) for seq in X_train]
    test_data_1 = [get_feature_1(seq) for seq in X_test]
    test_data_2 = [get_feature_2(seq) for seq in X_test]
    np.save("acp740trainf1.npy", train_data_1)
    np.save("acp740trainf2.npy", train_data_2)
    np.save("acp740testf1.npy", test_data_1)
    np.save("acp740testf2.npy", test_data_2)


# train_data_1 = [get_feature_1(seq) for seq in X_train]
# train_data_2 = [get_feature_2(seq) for seq in X_train]
# test_data_1 = [get_feature_1(seq) for seq in X_test]
# test_data_2 = [get_feature_2(seq) for seq in X_test]




train_data_1  =tensor(train_data_1, dtype=torch.float)
train_data_2  =tensor(train_data_2, dtype=torch.float)
test_data_1  =tensor(test_data_1, dtype=torch.float)
test_data_2  =tensor(test_data_2, dtype=torch.float)

tokenizer = AutoTokenizer.from_pretrained("preberts/smiles_tokenized")

train_data_3 = [Chem.MolToSmiles(Chem.MolFromSequence(seq)) for seq in X_train]
train_data_3 = tokenizer(train_data_3, padding=True, max_length=512, truncation=True, return_tensors="pt")
train_data_3 = train_data_3['input_ids'].to(torch.float)
print(train_data_1.shape, train_data_2.shape, train_data_3.shape)
train_data = torch.cat((train_data_1, train_data_2, train_data_3), dim=1)

test_data_3 = [Chem.MolToSmiles(Chem.MolFromSequence(seq)) for seq in X_test]
test_data_3 = tokenizer(test_data_3, padding=True, max_length=512, truncation=True, return_tensors="pt")
test_data_3 = test_data_3['input_ids'].to(torch.float)
print(test_data_1.shape, test_data_2.shape, test_data_3.shape)
test_data = torch.cat((test_data_1, test_data_2, test_data_3), dim=1)

X_train, y_train = torch.tensor(train_data), torch.tensor(y_train)
X_test, y_test = torch.tensor(test_data), torch.tensor(y_test)


# acp_seqs = acp_seqs + aug(acp_seqs) + aug_a(acp_seqs)
# nacp_seqs = nacp_seqs + aug(nacp_seqs) + aug_a(nacp_seqs)
#
# train_data = acp_seqs + nacp_seqs
# if os.path.exists("acp740f1.npy") and os.path.exists("acp740f2.npy"):
#     print('loading')
#     train_data_1 = np.load("acp740f1.npy")
#     train_data_2 = np.load("acp740f2.npy")
# else:
#     print('processing...')
#     print('acp740f1...')
#     train_data_1 = [get_feature_1(seq) for seq in train_data]
#     print('acp740f2...')
#     train_data_2 = [get_feature_2(seq) for seq in train_data]
#     np.save("acp740f1.npy", train_data_1)
#     np.save("acp740f2.npy", train_data_2)





# train_data_1  =tensor(train_data_1, dtype=torch.float)
# train_data_2  =tensor(train_data_2, dtype=torch.float)
# # tokenizer = AutoTokenizer.from_pretrained("preberts/chemberta_base")
# tokenizer = AutoTokenizer.from_pretrained("preberts/smiles_tokenized")
# # tokenizer = AutoTokenizer.from_pretrained("preberts/smiles_BPE")
# # bert_model = AutoModelForMaskedLM.from_pretrained("preberts/chemberta_base")
# train_data_3 = [Chem.MolToSmiles(Chem.MolFromSequence(seq)) for seq in train_data]
# train_data_3 = tokenizer(train_data_3, padding=True, max_length=512, truncation=True, return_tensors="pt")
# train_data_3 = train_data_3['input_ids'].to(torch.float)
# print(train_data_1.shape, train_data_2.shape, train_data_3.shape)
# # dtype=torch.int64
# train_data = torch.cat((train_data_1, train_data_2, train_data_3), dim=1)
# train_data = train_data.numpy()
# # print(sum(sum(train_data[:, 944:] != train_data_2)))
# train_label = [1] * len(acp_seqs) + [0] * len(nacp_seqs)
# print('split...')
# X_train, X_test, y_train, y_test =train_test_split(train_data, train_label, test_size=0.2, random_state=5)
# X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
# X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)



epochs = 50
model = acplstm(512, 512).to(device)
bert_params = list(map(id, model.bert.parameters()))
other_params = filter(lambda p: id(p) not in bert_params, model.parameters())
# optimizer = torch.optim.AdamW([{"params":other_params}, {"params":model.bert.parameters(),"lr":1e-4}], lr=1e-3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
hidden = None
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    time_start = time.time()
    hidden = train(train_dataloader, model, loss_fn, optimizer, hidden)
    time_end = time.time()
    print(f"train time: {(time_end-time_start)}")
    test(test_dataloader, model, None)
print("Done!")

# exit(0)s



# for batch, (X, y) in enumerate(train_dataloader):
#     pred, hidden = model(X, hidden)
#     print(pred)
#     loss = loss_fn(pred, y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()


# size = len(test_dataloader.dataset)
# model.eval()
# test_loss, correct = 0, 0
# y_score = []
# y_true = []
# with torch.no_grad():
#     for X, y in test_dataloader:
#         pred, _ = model(X)
#         print(pred.shape)
#         print(y.shape)
#         for i in range(16):
#             y_score.append(pred[i][y[i]])
#             y_true.append(y[i])
#         test_loss += loss_fn(pred, y).item()
#         correct += (pred.argmax(1) == y).to(torch.float).sum().item()
# test_loss /= size
# correct /= size
# print(f"correct = {correct}, Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

