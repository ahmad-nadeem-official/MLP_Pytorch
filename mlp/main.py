import torch
from torch import nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('cuda')
else:
    print('cpu')

x = torch.tensor(train.drop('price_range', axis = 1).values, dtype=torch.float32).to(device)
y = torch.tensor(train['price_range'].values, dtype=torch.long).to(device)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

class customDataset(Dataset):
  def __init__(self, x,y):
    self.x = x
    self.y = y
  def __len__(self):
    return len(self.x)
  def __getitem__(self, index):
    return self.x[index], self.y[index]

    #once you have made tensors don't use iloc etc
    #I had tenors above but i mistake here and used iloc

training_data = customDataset(x_train, y_train)
test_data = customDataset(x_test, y_test)

training_data = DataLoader(training_data, batch_size=32, shuffle=True)
test_data = DataLoader(test_data, batch_size=32, shuffle=True)

class simplenn(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 4)  # 4 classes
        )
    def forward(self, x):
        return self.network(x)
    

learning_rate = 0.01
epoches = 50

model = simplenn(x.shape[1]).to(device)

criterion = nn.CrossEntropyLoss() #loss fnx
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) #optemizer

for epoch in range(epoches):
  total_loss = 0
  for x_batch, y_batch in training_data:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    y_prd = model(x_batch)
    loss = criterion(y_prd, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
  print(f'epoch: {epoch+1}, loss: {total_loss/len(training_data)}')


model.eval()

with torch.no_grad():
  for x_batch, y_batch in test_data:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    y_prd = model(x_batch)
    _, y_pred = torch.max(y_prd.data, 1)
    total += y_batch.size(0)
    correct += (y_pred == y_batch).sum().item()
print((correct/total)*100)

# accuracy 93%

