import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

train = pd.read_csv("train_ctrUa4K.csv")
test = pd.read_csv("test_lAUu6dG.csv")

train_original = train.copy()
test_original = test.copy()

train.head()
test.head()

train.isnull().sum()

train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)

train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

train.isnull().sum()

train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

train.isnull().sum()

test.isnull().sum()

test['Gender'].fillna(test['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(test['LoanAmount'].median(), inplace=True)

test.isnull().sum()

train = train.drop('Loan_ID', axis=1)
test = test.drop('Loan_ID', axis=1)

x = train.drop('Loan_Status', 1)
y = train.Loan_Status

x = pd.get_dummies(x)
train = pd.get_dummies(train)
test = pd.get_dummies(test)

x_train, x_cv, y_train, y_cv = train_test_split(x,y,test_size=0.3)

modelz = GradientBoostingRegressor(random_state=0)

