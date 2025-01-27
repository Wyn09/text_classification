import pandas as pd
import torch 
import torch.nn as nn
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Iterable, Iterator
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter

from transformers import get_cosine_schedule_with_warmup


# tensorboard跟踪记录实现
# 1.SummaryWriter全局对象
# 2.跟踪相关:
#         title: train_loss,val_acc
#         value: loss，acc
#         indexer: train_cnt，acc_cnt

"""
    动态学习率
    混合精度
"""

class SummaryWrapper:
    def __init__(self):
        self.writer = SummaryWriter(r"F:\VsConde-Python\chen\source\logs")
        self.train_loss_cnt = 0

    def train_loss(self, func):
        def warpper(model, tokens, lbl, loss_fn, device):
            loss = func(model, tokens, lbl, loss_fn, device)
            self.writer.add_scalar("train loss", loss, self.train_loss_cnt)
            self.train_loss_cnt += 1
            return loss
        return warpper    

sw = SummaryWrapper()


class CommentDataset:
    def __init__(self, comments, labels, vocab):
        self.comments = comments
        self.labels = labels
        self.vocab = vocab 
    
        # 字典构建 (字符为token / 词汇为token)

    def __getitem__(self, index):
        token_index = [self.vocab.get(tk, self.vocab["<UNK>"]) for tk in self.comments[index]]
        index_tensor = torch.zeros(size=(125,), dtype=torch.long)
        for i in range(len(token_index)):
            index_tensor[i] = token_index[i]
        return index_tensor, torch.tensor(self.labels[index], dtype=torch.long)

    def __len__(self):
        return len(self.labels)
    

class Vocabulary:
    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def from_documents(cls, documents):
        tokens = set() 
        for cmt in documents:
            tokens.update(list(cmt))
        tokens = ["<PAD>", "<UNK>"] + sorted(list(tokens)) # set是无序的，可以在list之后做排序,保证每次构建词典顺序一致
        vocab = {token:i for i, token in enumerate(tokens)} 
        return cls(vocab)


def train_test_split(X, y , split_rate=0.2):
    """
    拆分数据集：
    1.拆分比率
    2.样本随机性
    3.构建拆分索引
    4.借助slice拆分
    """
    split_rate= 0.2
    split_size = int((len(X)) * (1 - split_rate))
    split_index = list(range(len(X)))
    random.shuffle(split_index)
    X_train = [X[i] for i in split_index[:split_size]]
    y_train = [y[i] for i in split_index[:split_size]]
    X_test = [X[i] for i in split_index[split_size:]]
    y_test = [y[i] for i in split_index[split_size:]]

    return X_train, X_test, y_train, y_test

def train(model, loss_fn, optimizer, train_dl, test_dl, epoch, device):

    train_steps = len(train_dl) * epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_training_steps=train_steps, num_warmup_steps=100)

    scaler = torch.amp.GradScaler(device=device)

    print(device)
    for e in range(epoch):
        model.train()
        process_bar = tqdm(train_dl)
        for tokens, lbl in process_bar:
            loss = train_step(model, tokens, lbl, loss_fn, device)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            process_bar.set_description(f"epoch: {e + 1}, lr: {scheduler.get_last_lr()[0]}, loss: {loss.item():.4f}")
        
        model.eval()
        with torch.no_grad():
            accuracy = 0
            for tokens, lbl in test_dl:
                tokens, lbl = tokens.to(device), lbl.to(device)   
                y_hat = model(tokens)
                accuracy += (y_hat.argmax(-1) == lbl).sum()
            print(f"accuracy: {(accuracy *100 / len(test_ds)):.4f}%")

@sw.train_loss
def train_step(model, tokens, lbl, loss_fn, device):
    tokens, lbl = tokens.to(device), lbl.to(device)
    # 混合精度前向计算
    with torch.amp.autocast(device):
        y_hat = model(tokens)

    loss = loss_fn(y_hat, lbl)
    return loss

class CommentsClassifier(nn.Module):
    def __init__(self, vocab_szie, embedding_size, rnn_hidden_size, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_szie, embedding_size, padding_idx=0)
        self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=rnn_hidden_size, batch_first=True)
        self.classifier = nn.Linear(rnn_hidden_size, num_labels)

    def forward(self, X):
        out = self.emb(X) # (batch_size, seq_len, embedding_size)
        output,_ = self.rnn(out) # (batch_size, seq_len, rnn_hidden_size)
        return self.classifier(output[:,-1,:]) # (batch_size, num_labels)
        pass

if __name__ == "__main__":
    batch_size = 64
    lr = 1e-2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch = 100
    embedding_size = 128
    rnn_hidden_size = 256
    num_labels = 2
   


    # 数据准备
    data = pd.read_pickle("./comments.bin")
    comments, labels = data["Comment"].values, data["labels"].values

    vocab = Vocabulary.from_documents(comments)

    # 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(comments, labels)
    train_ds = CommentDataset(X_train, y_train, vocab.vocab)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ds = CommentDataset(X_test, y_test, vocab.vocab)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)



    model = CommentsClassifier(
        vocab_szie=len(vocab.vocab), 
        embedding_size=embedding_size,
        rnn_hidden_size=rnn_hidden_size,
        num_labels=num_labels
    )
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    train(model, loss_fn, optimizer, train_dl, test_dl, epoch, device)


    save_path = "F:\VsConde-Python\chen\source\saved_models\model_objs.bin"
    torch.save(
        {
            "model_state":model.state_dict(),
            "model_vacob":vocab,
        },
        save_path
    )