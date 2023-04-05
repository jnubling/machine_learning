# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 12:19:29 2022

@author: Jonnathan Nubling
"""

import numpy as np
import matplotlib as plt
import pandas as pd

class DataPreprocessing():
    def __init__(self, path):
        # =============================================================================
        # importing the dataset
        # =============================================================================
        dataset = pd.read_csv(path)
        self.x_independent_variables = dataset.iloc[:, :-1].values
        self.y_dependent_variables = dataset.iloc[:, -1].values

        print('Importing the dataset')
        print(dataset)
        print("\n Dividing the dataset into:")
        print("independent variables")
        print(self.x_independent_variables)
        print("\ndependent variables")
        print(self.y_dependent_variables)

    def arrange_missing_data(self, inicial_column, final_column):
        # =============================================================================
        # taking care of missing data
        # =============================================================================
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        imputer.fit(self.x_independent_variables[:, inicial_column:final_column+1])
        self.x_independent_variables[:, inicial_column:final_column+1] = imputer.transform(
                                                        self.x_independent_variables[:, inicial_column:final_column+1]
                                                        )

        print('\ntaking care of missing data')
        print(self.x_independent_variables)

    def encode_categorical_independent_variable(self, column):
        # =============================================================================
        # encoding categorical data (independent variables)
        # =============================================================================
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

        ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [column])],
                                remainder='passthrough')

        self.x_independent_variables = np.array(ct.fit_transform(self.x_independent_variables))

        print('\nEncoding categorical data (independent variables)')
        print(self.x_independent_variables)

    def encode_categorical_dependent_variable(self):
        # =============================================================================
        # encoding categorical data (dependent variables)
        # =============================================================================
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        self.y_dependent_variables = le.fit_transform(self.y_dependent_variables)

        print("\nencoding categorical data (dependent variables)")
        print(self.y_dependent_variables)

    def split_training_and_test(self, test_size, random_state):
        # =============================================================================
        # splitting the dataset into training set and test set
        # =============================================================================
        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x_independent_variables,
                                                            self.y_dependent_variables,
                                                            test_size=test_size, # how will be the size of the test set, it means that it will take 20% of the dataset to be part only of the test set
                                                            random_state=random_state # seed variable
                                                            )
        print("\nsplitting the dataset into training set and test set:")
        print("independent variables training set")
        print(self.X_train)
        print("\nindependent variables test set")
        print(self.X_test)
        print("\ndependent variables training set")
        print(self.y_train)
        print("\ndependent variables test set")
        print(self.y_test)

    def feature_scaling(self):
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
        self.X_train[:, 3:] = sc.fit_transform(self.X_train[:, 3:])
        # perform standardization by centering and scaling all the variables
        self.X_test[:, 3:] = sc.transform(self.X_test[:, 3:])

        print("\nFeature Scaling")
        print("x trainig independent variables standardized")
        print(self.X_train)
        print("x test independent variables standardized")
        print(self.X_test)

