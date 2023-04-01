"""
Decision Tree Algorithm for regression model
One simple continuous value to predict (one dependent variable)
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import DecisionTreeRegressor


dataset = pd.read_csv('../data/Position_Salaries.csv')

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Visualising the data
plt.scatter(x=X, y=y, color='blue')
plt.title('Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Training the decision tree regression model on the whole dataset
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X=X, y=y)

# Predicting a new result
regressor.predict([[6.5]])

# Visualising prediction results in high resolution
X_grid = np.arange(
    min(X),
    max(X),
    0.1
    )
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(
    x=X,
    y=y,
    color='blue'
    )
plt.plot(X_grid, regressor.predict(X=X_grid), color='red')
plt.title('Decision Tree Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
