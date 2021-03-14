# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import os
import shutil
from optparse import OptionParser
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set(style='white', palette='deep')

from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_boston
boston = load_boston()

parser = OptionParser()
parser.add_option("-f", "--folder", dest="folder", default ="ujjwal_output")
(options, args) = parser.parse_args()

folder_name = options.folder
if os.path.isdir(folder_name):
  shutil.rmtree(folder_name)
os.mkdir(folder_name)

data = np.c_[boston['data'], boston['target']]
columns = np.append(boston['feature_names'], 'MEDV')

df= pd.DataFrame(data, columns= columns)

df.describe().loc[['min','max']]

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(df['MEDV'], bins=30)
plt.savefig(folder_name + '/distplot.png')

correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
plt.savefig(folder_name + '/heatmap.png')

"""LSTAT and RM are strongly correlated to Target variable MEDV."""

plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = df['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = df[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')

"""Linear Regression"""

lr_results = pd.DataFrame([],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])

for i in boston['feature_names']:
  X = df[[i]]
  y = df['MEDV']
  X_train1, X_test1, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
  X_train1.shape,X_test1.shape,y_train.shape,y_test.shape
  regressor = LinearRegression()
  regressor.fit(X_train1, y_train)
  coeff = regressor.coef_
  intercept = regressor.intercept_
  # Predicting Test Set
  y_pred = regressor.predict(X_test1)
  
  mae = metrics.mean_absolute_error(y_test, y_pred)
  mse = metrics.mean_squared_error(y_test, y_pred)
  rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
  r2 = metrics.r2_score(y_test, y_pred)
  model_results = pd.DataFrame([['Feature used : '+i, mae, mse, rmse, r2]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])
  # print(model_results)
  lr_results = lr_results.append(model_results, ignore_index = True)

results = lr_results.sort_values(by='RMSE', ascending=True)
print(results)
results.to_csv(folder_name + "/linear_regression.csv")

"""**Best feature with lowest RMSE is LSTAT followed by RM**"""

X = df[['LSTAT']]
y = df['MEDV']
X_train1, X_test1, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train1.shape,X_test1.shape,y_train.shape,y_test.shape
regressor = LinearRegression()
regressor.fit(X_train1, y_train)
coeff = regressor.coef_
intercept = regressor.intercept_

# plt.scatter(X_train1.AGE, y_train)
x1 = np.linspace(np.min(X_test1), np.max(X_test1))
y1 = coeff*x1 + intercept

plt.figure(figsize=(6,6))
plt.scatter(X_test1, y_test, label='Data Points')

plt.plot(x1, y1, linewidth=2.5, color='k', label='Best Fit Line')
plt.savefig(folder_name + '/linearRegression.png')

"""Polynomial Regression"""

X = df[['LSTAT']]
y = df['MEDV']
X_train1, X_test1, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train1.shape,X_test1.shape,y_train.shape,y_test.shape
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train1)
regressor = LinearRegression()
regressor.fit(X_poly, y_train)
coeff = regressor.coef_
intercept = regressor.intercept_
## Predicting test values
y_pred = regressor.predict(poly_reg.fit_transform(X_test1))

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)

results = pd.DataFrame([['Polynomial Linear Regression', mae, mse, rmse, r2]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])
results

new_X = np.linspace(X_test1.min(),X_test1.max())

new_y = regressor.predict(poly_reg.fit_transform(new_X))

plt.figure(figsize=(6,6))
plt.scatter(X_test1['LSTAT'], y_test, label='Data Points')
plt.plot(new_X,new_y)
plt.savefig(folder_name + '/polynomialRegression.png')

poly_reg_20 = PolynomialFeatures(degree = 20)
X_poly_20 = poly_reg_20.fit_transform(X_train1)
regressor = LinearRegression()
regressor.fit(X_poly_20, y_train)
coeff = regressor.coef_
intercept = regressor.intercept_
## Predicting values
y_pred = regressor.predict(poly_reg_20.fit_transform(X_test1))

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)

poly_result_20 = pd.DataFrame([['Polynomial Linear Regression Degree=20', mae, mse, rmse, r2]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])
results = results.append(poly_result_20, ignore_index=True)
results

new_y_20 = regressor.predict(poly_reg_20.fit_transform(new_X))

plt.figure(figsize=(6,6))
plt.scatter(X_test1['LSTAT'], y_test, label='Data Points')
plt.plot(new_X,new_y_20)
plt.savefig(folder_name + '/polynomialRegression20.png')

"""### Multiple Linear Regression"""

X = df.loc[:,boston['feature_names']]
y = df.loc[:,'MEDV']

X_train1, X_test1, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train1.shape,X_test1.shape,y_train.shape,y_test.shape

regressor = LinearRegression()
regressor.fit(X_train1, y_train)

# Predicting Test Set
y_pred = regressor.predict(X_test1)

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2)*(len(y_test)-1/len(y_test)-len(boston['feature_names'])-1)
multivariate = pd.DataFrame([['Multiple Linear Regression', mae, mse, rmse, r2,adj_r2]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score', 'Adjusted R2 score'])
results = results.append(multivariate, ignore_index=True)
print(results)
results.to_csv(folder_name + "/final_results.csv")