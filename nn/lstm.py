import pandas as pd
import keras
import pickle as pkl
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import  mean_squared_error, mean_absolute_error, r2_score,accuracy_score
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

n_input = 1
n_features = X_train.shape[1] 

generator = TimeseriesGenerator(X_train, y_train, length=n_input, batch_size=32)

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1)) 
model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())

model.summary()

# Train the model
model.fit(generator, epochs=5)


# with open('lstm_model.pkl', 'wb') as file:
#     pkl.dump(model, file)


# Prepare X_test for prediction
test_generator = TimeseriesGenerator(X_test, y_test, length=n_input, batch_size=32)


y_pred = model.predict(test_generator)

print("Predictions:", y_pred)

mse = mean_squared_error(y_test[n_input:], y_pred)  # Ignore the first n_input samples
mae = mean_absolute_error(y_test[n_input:], y_pred)
r2 = r2_score(y_test[n_input:], y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-Squared: {r2:.4f}")

threshold = 0.5  
y_pred_binary = (y_pred > threshold).astype(int)
accuracy = accuracy_score(y_test[n_input:], y_pred_binary)

print(f"Accuracy: {accuracy:.4%}")