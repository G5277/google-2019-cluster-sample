import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, Lasso, LassoLars, ElasticNet, Lars, HuberRegressor,OrthogonalMatchingPursuit, PassiveAggressiveRegressor

models = {
    "dt_model": DecisionTreeRegressor(),
    "rf_model": RandomForestRegressor(),
    "et_model": ExtraTreesRegressor(),
    #"gb_model": GradientBoostingRegressor(),
    # "ab_model": AdaBoostRegressor(),
    # "lr_model": LinearRegression(),
    # "rr_model": Ridge(),
    # "br_model": BayesianRidge(),
    # "lasso_model": Lasso(),
    # "lla_model": LassoLars(),
    # "en_model": ElasticNet(),
    # "la_model": Lars(),
    # "hb_model": HuberRegressor(),
    # "knn_model": KNeighborsRegressor(), --> 67%
    # "omp_model": OrthogonalMatchingPursuit(),
    # "pa_model": PassiveAggressiveRegressor(),
    # "dm_model": DummyRegressor(),
    "xgb_model": XGBRegressor(),
    # "lgb_model": LGBMRegressor()
}

data = pd.read_csv('./data/regression_data.csv')
data = data.dropna()
X = data.drop('exec_time', axis = 1)
y = data['exec_time']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

for m_name in models:
    print(F"Model {m_name}")
    model = regressors[m_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = r2_score(y_pred, y_test)
    print(f"Model {m_name} has accuracy {acc}.")
    print("               ")