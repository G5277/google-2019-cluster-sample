import pandas as pd
data = pd.read_csv('./data/clean.csv')
print(data.corr()['failed'])