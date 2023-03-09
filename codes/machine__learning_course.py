# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 12:19:29 2022

@author: Jonnathan Nubling
"""

import numpy as np
import matplotlib as plt
import pandas as pd


# =============================================================================
# importing the dataset
# =============================================================================
dataset = pd.read_csv("../data/Data.csv")
x_independent_variables = dataset.iloc[:, :-1].values
y_dependent_variables = dataset.iloc[:, -1].values

print('Importing the dataset')
print(dataset)
print("\n Dividing the dataset into:")
print("independent variables")
print(x_independent_variables)
print("\ndependent variables")
print(y_dependent_variables)

# =============================================================================
# taking care of missing data
# =============================================================================
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x_independent_variables[:, 1:3])
x_independent_variables[:, 1:3] = imputer.transform(
                                                x_independent_variables[:, 1:3]
                                                )

print('\ntaing care of missing data')
print(x_independent_variables)

# =============================================================================
# encoding categorical data (independent variables)
# =============================================================================
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])],
                        remainder='passthrough')

x_independent_variables = np.array(ct.fit_transform(x_independent_variables))

print('\nEncoding categorical data (independent variables)')
print(x_independent_variables)

# =============================================================================
# encoding categorical data (dependent variables)
# =============================================================================
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_dependent_variables = le.fit_transform(y_dependent_variables)

print("\nencoding categorical data (dependent variables)")
print(y_dependent_variables)

# =============================================================================
# splitting the dataset into training set and test set
# =============================================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_independent_variables,
                                                    y_dependent_variables,
                                                    test_size=0.2, # how will be the size of the test set, it means that it will take 20% of the dataset to be part only of the test set
                                                    random_state=1 # seed variable
                                                    )
print("\nsplitting the dataset into training set and test set:")
print("independent variables training set")
print(X_train)
print("\nindependent variables test set")
print(X_test)
print("\ndependent variables training set")
print(y_train)
print("\ndependent variables test set")
print(y_test)

# =============================================================================
# feature scaling
# =============================================================================
""" 
feature scaling will not be applied in all the machine learning models,
in general it is only applied in gradient descedent based algorithms, witch 
includes logistic regretion and neural networks.
Ps.: FS must always be applied after splitting the dataset into test set 
and training set
 """
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:]) 
# perform standardization by centering and scaling all the variables
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print("\nFeature Scaling")
print("x trainig independent variables standardized")
print(X_train)
print("x test independent variables standardized")
print(X_test)

