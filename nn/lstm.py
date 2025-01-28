import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# Data
data = pd.read_csv('./data/clean.csv')
X = data.drop('failed', axis = 1)
y = data['failed']

# Split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 0)

# Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# print(X_train.shape[1])
generator = TimeseriesGenerator(X_train.drop('failed', axis = 1), X_train['failed'], 1, batch_size = 1)
X, y = generator
print(f'Given the Array: \n{X[0].flatten()}')
print(f'Predict this y: \n {y[0]}')