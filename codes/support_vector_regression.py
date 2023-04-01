"""
Support Vector Regression (SVR) Algorithm
One simple continuous value to predict (one dependent variable)
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


dataset = pd.read_csv('../data/Position_Salaries.csv')

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Visualising the data
plt.scatter(x=X, y=y, color='blue')
plt.title('Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X=X)
y = y.reshape(len(y),1)
y = sc_y.fit_transform(y)

# Training the SVR Model on the whole dataset
regressor = SVR(kernel='rbf')
regressor.fit(X=X, y=y)

# Predicting new values
sc_y.inverse_transform(
    regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1)
    )

# Visualising the SVR results
plt.scatter(
    x=sc_X.inverse_transform(X),
    y=sc_y.inverse_transform(y),
    color='blue'
    )
plt.plot(
    sc_X.inverse_transform(X),
    sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)),
    color='red'
    )
plt.title('Support Vector Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Ajusting for higher resolution and smoothness
X_grid = np.arange(
    min(sc_X.inverse_transform(X)),
    max(sc_X.inverse_transform(X)),
    0.1
    )
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(
    x=sc_X.inverse_transform(X),
    y=sc_y.inverse_transform(y),
    color='blue'
    )
plt.plot(
    X_grid,
    sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)),
    color='red'
    )
plt.title('Support Vector Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()