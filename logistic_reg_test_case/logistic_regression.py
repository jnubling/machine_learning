"""
Logistic Regression Model
for breast cancer prediction
"""
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix


# import the dataset
dataset = pd.read_csv('breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# split into test and training sets
X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y,
                                        test_size=0.2,
                                        random_state=0
                                    )

# train the logistic regression model
classifier = LogisticRegression(random_state=0)
classifier.fit(X=X_train, y=y_train)

# predict the test set results
y_pred = classifier.predict(X=X_test)

# make the confusion matrix
cf = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(cf)

# compute the accuracy with k-fold cross validation
accuracies = cross_val_score(
                                estimator=classifier,
                                X=X_train,
                                y=y_train,
                                cv=10,
                            )
print('Accuracy: {:.2f}%'.format(accuracies.mean()*100))
print('Standard Deviation: {:.2f}%'.format(accuracies.std()*100))