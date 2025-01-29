from class_CNN import model_CNN
import pandas as pd
import os
import sys
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


if torch.cuda.is_available:
    device = torch.device('cuda:0')
else:
    torch.device = torch.device('cpu')

# Load data
data = pd.read_csv('./data/clean.csv')

# Split features and target
X = data.drop('failed', axis=1).values  # Convert to NumPy array
y = data['failed'].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Converting to torch tensors
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).long()
# y_test = torch.tensor(y_test)#.float()

# Reshaping for CNN fit
X_train = X_train.unsqueeze(1)

cnet = model_CNN(input_length=X_train.shape[2]).to(device) 
optimizer = torch.optim.Adam(cnet.parameters(), lr=1e-3)
loss_function = torch.nn.CrossEntropyLoss()

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)

print("start train")

def train_model():
    EPOCHS = 8
    BATCH = 50
    prev = float('inf')
    loss = sys.maxsize + 5
    for epoch in range(EPOCHS):
        if (abs(loss - prev) > 0.001):
            for i in tqdm(range(0, len(X_train), BATCH)):
                prev = loss
                batch_X = X_train[i:i+BATCH]
                # print(f"Size X {len(batch_X)}")
                batch_y = y_train[i:i+BATCH]
                # print(f"Size Y {len(batch_y)}")
                cnet.zero_grad()
                # print('1')
                output = cnet(batch_X)
                # print('2')
                # output = torch.argmax(output)
                # print('3')
                loss = loss_function(output, batch_y)
                loss.backward()
                optimizer.step()
        else:
            print(f"ending, {loss-prev}")
            break
        print(f"{epoch} : {loss}")

    with open("model.pt", "wb") as f:
        torch.save(cnet.state_dict(), f)
TRAIN = True
if TRAIN:
    train_model()


with open("./model.pt", "rb") as f:
    cnet.load_state_dict(torch.load(f), strict=True)

for i in range(0, len(X_test)):
    batch_x = X_test[i:i+1]
    # print(batch_y)
    output = cnet(batch_x)
    if (torch.argmax(output) == y_test[i]-1):
        win = win + 1
    else:
        print(torch.argmax(output), y_test[i]-1)
        lose = lose + 1

print(f"Accuracy {win/(win+lose)}")
print(f"{lose}")