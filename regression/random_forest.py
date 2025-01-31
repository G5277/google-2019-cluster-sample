import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('./data/regression_data.csv')

X = data.drop('exec_time', axis = 1)
y = data['exec_time']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


rf_model = RandomForestRegressor(n_estimators = 20, random_state = 0)

rf_model.fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_pred, y_test)

print(f"Acc : {acc}")