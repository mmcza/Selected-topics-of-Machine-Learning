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
from ydata_profiling import ProfileReport
import seaborn as sns
import scipy

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
    #print(diabetes_no_missing.data.describe())

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
    pregnancies = diabetes.data.to_numpy()[:, 0].reshape(-1, 1)
    #print(pd.DataFrame(np.concatenate((pregnancies, diabetes_imp_mean), axis=1)).describe())
    diabetes_imp_mean = pd.DataFrame(np.concatenate((pregnancies, diabetes_imp_mean), axis=1))
    diabetes_imp_mean.columns = diabetes.data.columns
    #print(diabetes_imp_mean.describe())

    X_train_im, X_test_im, y_train_im, y_test_im = train_test_split(diabetes_imp_mean, diabetes.target, test_size=0.2, random_state=42)

    # predict the target using the DecisionTreeClassifier
    imputed_mean_DTC_clf = DecisionTreeClassifier()
    imputed_mean_DTC_clf.fit(X_train_im.to_numpy(), y_train_im.to_numpy())
    pred_imputed_mean_DTC = imputed_mean_DTC_clf.predict(X_test_im.to_numpy())
    accuracy_imputed_mean_DTC = sklearn.metrics.accuracy_score(y_test_im, pred_imputed_mean_DTC)
    print("Decision Tree Classifier with imputed mean values had accuracy of: "+str(accuracy_imputed_mean_DTC))

    # predict the target using the RandomForestClassifier
    imputed_mean_RFC_clf = RandomForestClassifier()
    imputed_mean_RFC_clf.fit(X_train_im.to_numpy(), y_train_im.to_numpy())
    pred_imputed_mean_RFC = imputed_mean_RFC_clf.predict(X_test_im.to_numpy())
    accuracy_imputed_mean_RFC = sklearn.metrics.accuracy_score(y_test_im, pred_imputed_mean_RFC)
    print("Random Forest Classifier with imputed mean values had accuracy of: "+str(accuracy_imputed_mean_RFC))

    # predict the target using the SVC
    imputed_mean_SVC_clf = SVC(gamma='auto')
    imputed_mean_SVC_clf.fit(X_train_im.to_numpy(), y_train_im.to_numpy())
    pred_imputed_mean_SVC = imputed_mean_SVC_clf.predict(X_test_im.to_numpy())
    accuracy_imputed_mean_SVC = sklearn.metrics.accuracy_score(y_test_im, pred_imputed_mean_SVC)
    print("Support Vector Classifier with imputed mean values had accuracy of: "+str(accuracy_imputed_mean_SVC))

    # get profile report
    profile = ProfileReport(diabetes.data, title="Profiling Report")
    profile.to_file("your_report.html")

def task2():
    diabetes = fetch_openml("diabetes", version=1, as_frame=True)
    #histogram = diabetes.data.mass.hist()
    histogram = sns.histplot(data = diabetes.data, x = 'mass')
    plt.show()
    boxplot = sns.boxplot(data = diabetes.data, x = 'mass')
    plt.show()

def task3():
    diabetes = fetch_openml("diabetes", version=1, as_frame=True)
    plt.scatter(diabetes.data['plas'], diabetes.data['mass'])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Pima Indians Diabetes Database')
    plt.xlabel('plas')
    plt.ylabel('mass')
    plt.show()

def task4():
    diabetes = fetch_openml("diabetes", version=1, as_frame=True)
    #check z-score of the data
    # z_score = (diabetes.data - diabetes.data.mean())/diabetes.data.std()
    z_score = scipy.stats.zscore(diabetes.data)
    # save index of rows with z-score > 3
    outliers = np.where(np.abs(z_score) > 3)
    #print(outliers[0])
    # remove rows with z-score > 3
    diabetes.data = diabetes.data.drop(outliers[0])
    diabetes.target = diabetes.target.drop(outliers[0])
    #print(z_score)
    #new_z_score = scipy.stats.zscore(diabetes.data)
    #print(new_z_score)

    # other possible ways:
    # using sklearn.covariance.EllipticEnvelope

    # # using sklearn.svm.OneClassSVM
    # ocsvm = sklearn.svm.OneClassSVM(nu=0.2)
    # ocsvm.fit(diabetes.data)
    # # Predict the labels (1 for inliers, -1 for outliers)
    # labels = ocsvm.predict(diabetes.data)
    # outliers = np.where(np.array(labels) == -1)
    # diabetes.data = diabetes.data.drop(outliers[0])
    # diabetes.target = diabetes.target.drop(outliers[0])


    # using sklearn.ensemble.IsolationForest
    # using sklearn.neighbors.LocalOutlierFactor


    # draw scatter plot again
    plt.scatter(diabetes.data['plas'], diabetes.data['mass'])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Pima Indians Diabetes Database')
    plt.xlabel('plas')
    plt.ylabel('mass')
    plt.show()

if __name__ == '__main__':
    #task1()
    #task2()
    #task3()
    task4()