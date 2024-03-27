import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn import impute

pd.set_option('display.max_columns', None)

def task1():
    diabetes = fetch_openml("diabetes", version=1, as_frame=True)

    print(diabetes.data.describe())
    # print(diabetes.target)
    # divide the data into training and testing - select all data from data apart from last column
    X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=42)

    # predict the target using the DecisionTreeClassifier
    basic_DTC_clf = DecisionTreeClassifier()
    basic_DTC_clf.fit(X_train.to_numpy(), y_train.to_numpy())
    pred_basic_DTC = basic_DTC_clf.predict(X_test.to_numpy())
    accuracy_basic_DTC = sklearn.metrics.accuracy_score(y_test, pred_basic_DTC)
    print("Basic Decision Tree Classifier had accuracy of: "+str(accuracy_basic_DTC))

    # predict the target using the RandomForestClassifier
    basic_RFC_clf = RandomForestClassifier()
    basic_RFC_clf.fit(X_train.to_numpy(), y_train.to_numpy())
    pred_basic_RFC = basic_RFC_clf.predict(X_test.to_numpy())
    accuracy_basic_RFC = sklearn.metrics.accuracy_score(y_test, pred_basic_RFC)
    print("Basic Random Forest Classifier had accuracy of: "+str(accuracy_basic_RFC))

    # predict the target using the SVC
    basic_SVC_clf = SVC(gamma='auto')
    basic_SVC_clf.fit(X_train.to_numpy(), y_train.to_numpy())
    pred_basic_SVC = basic_SVC_clf.predict(X_test.to_numpy())
    accuracy_basic_SVC = sklearn.metrics.accuracy_score(y_test, pred_basic_SVC)
    print("Basic Support Vector Classifier had accuracy of: "+str(accuracy_basic_SVC))

    # remove missing values - remove rows where data from any column (apart from preg) is equal to 0
    # get column names
    diabetes_no_missing = diabetes
    for column in diabetes.data.columns:
        if column != 'preg' and column != 'age':
            diabetes_no_missing.target = diabetes_no_missing.target[~(diabetes_no_missing.data[column] == 0)]
            diabetes_no_missing.data = diabetes_no_missing.data[~(diabetes_no_missing.data[column] == 0)]
    print(diabetes_no_missing.data.describe())

    # divide the data into training and testing - select all data from data apart from last column
    X_train_nm, X_test_nm, y_train_nm, y_test_nm = train_test_split(diabetes_no_missing.data, diabetes_no_missing.target, test_size=0.2, random_state=42)

    # predict the target using the DecisionTreeClassifier
    removed_data_DTC_clf = DecisionTreeClassifier()
    removed_data_DTC_clf.fit(X_train_nm.to_numpy(), y_train_nm.to_numpy())
    pred_removed_data_DTC = removed_data_DTC_clf.predict(X_test_nm.to_numpy())
    accuracy_removed_data_DTC = sklearn.metrics.accuracy_score(y_test_nm, pred_removed_data_DTC)
    print("Decision Tree Classifier with removed data had accuracy of: "+str(accuracy_removed_data_DTC))

    # predict the target using the RandomForestClassifier
    removed_data_RFC_clf = RandomForestClassifier()
    removed_data_RFC_clf.fit(X_train_nm.to_numpy(), y_train_nm.to_numpy())
    pred_removed_data_RFC = removed_data_RFC_clf.predict(X_test_nm.to_numpy())
    accuracy_removed_data_RFC = sklearn.metrics.accuracy_score(y_test_nm, pred_removed_data_RFC)
    print("Random Forest Classifier with removed data had accuracy of: "+str(accuracy_removed_data_RFC))

    # predict the target using the SVC
    removed_data_SVC_clf = SVC(gamma='auto')
    removed_data_SVC_clf.fit(X_train_nm.to_numpy(), y_train_nm.to_numpy())
    pred_removed_data_SVC = removed_data_SVC_clf.predict(X_test_nm.to_numpy())
    accuracy_removed_data_SVC = sklearn.metrics.accuracy_score(y_test_nm, pred_removed_data_SVC)
    print("Support Vector Classifier with removed data had accuracy of: "+str(accuracy_removed_data_SVC))

    # impute missing values - replace missing values with mean
    imp_mean = impute.SimpleImputer(missing_values=0, strategy='mean')
    imp_mean.fit(diabetes.data.to_numpy()[:, 1:])
    diabetes_imp_mean = imp_mean.transform(diabetes.data.to_numpy()[:, 1:])
    #print(pd.DataFrame(np.concatenate((diabetes.data.to_numpy()[:, 0], diabetes_imp_mean), axis= 0)).describe())


if __name__ == '__main__':
    task1()