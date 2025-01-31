import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

data = pd.read_csv('./data/regression_data.csv')
data = data.dropna()

X = data.drop(['exec_time', 'req_cpus', 'req_memory'], axis=1)
y = data[['exec_time', 'req_cpus', 'req_memory']] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

models = {
    "dt_model": DecisionTreeRegressor(),
    "rf_model": RandomForestRegressor(),
    "et_model": ExtraTreesRegressor(),
    "xgb_model": XGBRegressor(),
}

for m_name, model in models.items():
    print(f"Training Model: {m_name}")

    mo_model = MultiOutputRegressor(model)
    mo_model.fit(X_train, y_train)
    y_pred = mo_model.predict(X_test)
    r2_scores = [r2_score(y_test.iloc[:, i], y_pred[:, i]) for i in range(y.shape[1])]
    print(f"Model {m_name} R2 Scores: {r2_scores}\n")
