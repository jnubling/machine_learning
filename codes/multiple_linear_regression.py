"""
Multiple Linear Regression Algorithm
One simple continuous value to predict (one dependent variable)
"""
from data_preprocessing import DataPreprocessing

from sklearn.linear_model import LinearRegression
import numpy as np


dataset = DataPreprocessing(path='../data/50_Startups.csv')

dataset.encode_categorical_independent_variable(column=3)

dataset.split_training_and_test(test_size=0.2, random_state=0)

###################################################################
# visualising the data
###################################################################


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