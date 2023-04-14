"""
Decision Tree Classification Model
One simple categorical probability to predict
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


dataset = pd.read_csv('C:/Users/jonna/Documents/Programming/machine_learning/data/Social_Network_Ads.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# visualising the data
plt.scatter(x=X[:, 0], y=y, color='blue')
plt.title('Social Network Ads')
plt.xlabel('Age')
plt.ylabel('Purchased')

# feature scaling
sc = StandardScaler()
X = sc.fit_transform(X=X)

# splitting dataset
X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y,
                                        test_size=0.25,
                                        random_state=0
                                        )

# train the Naive Bayes model on the training set
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# predicting a new value
print(classifier.predict(sc.transform([[30, 87000]])))

# predicting the Test set results
y_pred = classifier.predict(X=X_test)
print(
    np.concatenate(
        (
            y_pred.reshape(len(y_pred),1),
            y_test.reshape(len(y_test),1)
        ),
         1,
         ))

#making the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# visualising training set results
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(
                        start=X_set[:, 0].min() - 10,
                        stop=X_set[:, 0].max() + 10,
                        step=0.25
                        ),
                     np.arange(
                        start=X_set[:, 1].min() - 1000,
                        stop=X_set[:, 1].max() + 1000,
                        step=0.25
                        )
                    )
plt.contourf(
    X1,
    X2,
    classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green')),
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
            X_set[y_set == j, 0],
            X_set[y_set == j, 1],
            c=ListedColormap(('red', 'green'))(i),
            label=j
        )
plt.title('Naïve Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# visualising test set results
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(
                        start=X_set[:, 0].min() - 10,
                        stop=X_set[:, 0].max() + 10,
                        step=0.25
                        ),
                     np.arange(
                        start = X_set[:, 1].min() - 1000,
                        stop=X_set[:, 1].max() + 1000,
                        step=0.25
                        )
                    )
plt.contourf(
    X1,
    X2,
    classifier.predict(
                sc.transform(
                        np.array(
                            [X1.ravel(), X2.ravel()]
                            ).T
                        )
                ).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
    )
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(('red', 'green'))(i),
        label=j
        )
plt.title('Naïve Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()