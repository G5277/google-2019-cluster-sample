import pandas as pd
import pycaret
from pycaret.classification import *

print(pycaret.__version__)
data = pd.read_csv('./data/clean.csv')

s = setup(data = data, target='failed')
cm = compare_models()