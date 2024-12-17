import torch
import torch.nn as nn
import torch.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from model.LSTMmodel import LSTMModel

import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="path to input csv file", default="data/bodo_poopy.csv")
parser.add_argument("-e", "--epoch", type=int, help="number of epochs", default=10)
parser.add_argument("-lr", "--learning_rate", type=float, help="learning rate", default=0.001)
parser.add_argument("-emb", "--embedding_dim", type=int, help="embedding dimension", default=64)
args = parser.parse_args()

lr = args.learning_rate
embedding_dim = args.embedding_dim
epoch = args.epoch
input_file = args.input

df = pd.read_csv(input_file, index_col=0)

X, y = df["text"], df["Label"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)

def prepare_sequence(seqs, to_ix):
    idxs = []
    maxl = 0
    for seq in seqs:
       out = [to_ix.get(w, 0) for w in seq]
       maxl = max(maxl, len(out))
       idxs.append(out)
    idxs = [i + [0]*(maxl-len(i)) for i in idxs]
    return torch.tensor(idxs, dtype=torch.long)

vocabulary = list(set(X_train.str.cat(sep=' ').split()))
word_to_ix = {word: i+1 for i, word in enumerate(vocabulary)}
ix_to_word = {i: word for word, i in word_to_ix.items()}
ix_to_word[0] = "[OOV]"

ohe_encoder = OneHotEncoder(sparse_output=False)
y_train = ohe_encoder.fit_transform(y_train)
y_test  = ohe_encoder.transform(y_test)

X_train = prepare_sequence(X_train, word_to_ix)
X_test = prepare_sequence(X_test, word_to_ix)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

torch.manual_seed(42)

trainloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train),
    batch_size=64, 
    shuffle=True
)

testloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_test, y_test),
    batch_size=64, 
    shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMModel(len(word_to_ix), embedding_dim, 128, 2, 2)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()

model.to(device)

print("Training...")

for e in range(epoch):
    acc_loss = 0.0
    acc_score = 0.0
    for i, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_loss += loss.item()
        acc_score += (output.argmax(1) == y.argmax(1)).float().mean().item()
        
    print(f"Epoch: {e+1}, Loss: {acc_loss/(i+1)}, Accuracy: {acc_score/(i+1):0.4f}")

print("."*10,"\n")
print("Testing...")

acc_loss = 0.0
acc_score = 0.0
model.eval()
with torch.no_grad():
    for i, (x, y) in enumerate(testloader):
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output, y)

        acc_loss += loss.item()
        acc_score += (output.argmax(1) == y.argmax(1)).float().mean().item()
    
print(f"Loss: {acc_loss/(i+1)}, Accuracy: {acc_score/(i+1):0.4f}")

    