import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

import warnings

train = pd.read_csv("")
test = pd.read_csv("")

test.head()
train_original=train.copy()
test_original=test.copy()

train.isnull().sum()
