import torch
from torch import load
from class_CNN import model_CNN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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
# X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
# y_train = torch.tensor(y_train).long()
y_test = torch.tensor(y_test)

cnet = model_CNN()

with open("./model.pt", "rb") as f:
    cnet.load_state_dict(load(f), strict=True)

win = 0
lose = 0
for i in range(0, len(X_test)):
    batch_y = X_test[i:i+1]
    # print(batch_y)
    output = cnet(batch_y)
    if (torch.argmax(output) == y_test[i]-1):
        win = win + 1
    else:
        print(torch.argmax(output), y_test[i]-1)
        lose = lose + 1


print(f"Accuracy {win/(win+lose)}")
print(f"{lose}")