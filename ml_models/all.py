import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = pd.read_csv('./data/clean.csv')
X = data.drop('failed', axis = 1)
y = data['failed']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)



models = {
    'rf_model' : RandomForestClassifier(),
    'et_model' : ExtraTreesClassifier(),
    'ada_model' : AdaBoostClassifier(n_estimators = 100, random_state = 0),
    'cat_model' : CatBoostClassifier(),
    'xgb_model' : XGBClassifier() 
}

for m_name in models:
    model = models[m_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    print(f"Model {m_name} has accuracy {acc}.")