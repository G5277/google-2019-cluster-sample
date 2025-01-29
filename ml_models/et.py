import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = pd.read_csv('./data/clean.csv')
X = data.drop('failed', axis = 1)
y = data['failed']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

et_model = ExtraTreesClassifier(n_estimators = 2, random_state = 0)
et_model.fit(X_train, y_train)

y_pred = et_model.predict(X_test)

# Eval
acc = accuracy_score(y_pred,y_test)
print(f"Acc {acc}") 