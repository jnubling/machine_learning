"""
Multiple Linear Regression Algorithm
One simple continuous value to predict (one dependent variable)
"""
from data_preprocessing import DataPreprocessing

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import numpy as np
import matplotlib.pyplot as plt


dataset = DataPreprocessing(path='../data/50_Startups.csv')

dataset.encode_categorical_independent_variable(column=3)

dataset.split_training_and_test(test_size=0.2, random_state=0)

###################################################################
# visualising the data
###################################################################
plt.scatter(
    x=dataset.x_independent_variables[:,3],
    y=dataset.y_dependent_variables,
    color='blue')
plt.title('R&D Spend vs Profit')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.show()

plt.scatter(
    x=dataset.x_independent_variables[:,4],
    y=dataset.y_dependent_variables,
    color='blue')
plt.title('Administration Spend vs Profit')
plt.xlabel('Administration Spend')
plt.ylabel('Profit')
plt.show()

plt.scatter(
    x=dataset.x_independent_variables[:,5],
    y=dataset.y_dependent_variables,
    color='blue')
plt.title('Marketing Spend vs Profit')
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.show()

###################################################################
# training the multiple linear regression model on the test set
###################################################################
regressor = LinearRegression()
regressor.fit(X=dataset.X_train, y=dataset.y_train)

###################################################################
# predicting the multiple linear regression model on the test set
###################################################################
y_predicted = regressor.predict(dataset.X_test)

###################################################################
# Visualising
# #################################################################
np.set_printoptions(precision=2)
print('Printing predicted_values[0] vs real_values[1]')
print(
    np.concatenate(
        (
            y_predicted.reshape(len(y_predicted), 1),
            dataset.y_test.reshape(len(dataset.y_test), 1)
        ),
        1)
    )

###################################################################
# Predicting a single value
###################################################################
# profit of a startup with R&D Spend = 160000,
# Administration Spend = 130000,
# Marketing Spend = 300000 and State = 'California'(1,0,0)
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

# evaluating the model performance
# r2_score(y_true=y_test, y_pred=y_pred)
