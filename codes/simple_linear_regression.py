"""
Simple Linear Regression Algorithm
One simple continuous value to predict (one dependent variable)
"""
from data_preprocessing import DataPreprocessing

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

dataset = DataPreprocessing(path='../data/Salary_Data.csv')

dataset.split_training_and_test(test_size=0.2, random_state=0)

###################################################################
# visualising the data
###################################################################
plt.scatter(x=dataset.x_independent_variables, y=dataset.y_dependent_variables, color='red')
plt.title('Salary vs Experience')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

###################################################################
# training the simple linear regression model on the test set
###################################################################
regressor = LinearRegression()
regressor.fit(X=dataset.X_train, y=dataset.y_train)

###################################################################
# predicting the simple linear regression model on the test set
###################################################################
y_predicted = regressor.predict(X=dataset.X_test)

###################################################################
# Visualising the training set result
# #################################################################
plt.scatter(x=dataset.X_train, y=dataset.y_train, color='red')
plt.plot(dataset.X_train, regressor.predict(X=dataset.X_train), color='blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

###################################################################
# Visualising the test set result
###################################################################
plt.scatter(x=dataset.X_test, y=dataset.y_test, color='red')
plt.plot(dataset.X_train, regressor.predict(X=dataset.X_train), color='blue')
plt.title('Salary vs Experience (test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

###################################################################
# Predicting a single value and getting the coefficients
###################################################################
print(f'A person with 12 years of experience will have a salary of ${regressor.predict([[12]])[0]}.')

print(f'\nThe coefficient b0 of the equation is {regressor.coef_}')
print(f'The point of interception is {regressor.intercept_}')

print(f'\n The final equation is: Salary = {regressor.coef_[0]} X Years of Experience + {regressor.intercept_}')