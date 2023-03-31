"""
Polynomial Regression Algorithm
One simple continuous value to predict (one dependent variable)
"""
# from data_preprocessing import DataPreprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('../data/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

plt.scatter(x=X, y=y, color='blue')
plt.title('Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X=X)
lin_reg = LinearRegression()
lin_reg.fit(X=X_poly, y=y)

plt.scatter(x=X, y=y, color='blue')
plt.plot(X, lin_reg.predict(X=X_poly), color='red')
plt.title('Polynomial Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

lin_reg.predict(X=poly_reg.fit_transform(X=[[6.5]]))