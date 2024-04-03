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
from sklearn import svm, model_selection
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import RidgeCV
import shap
import xgboost as xgb
from xgboost import plot_importance

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

def task5():
    diabetes = fetch_openml("diabetes", version=1, as_frame=True)
    # remove outliers using IsolationForest
    Iso_Forest = sklearn.ensemble.IsolationForest(contamination=0.3)
    Iso_Forest.fit(diabetes.data)
    labels_Iso_Forest = Iso_Forest.predict(diabetes.data)
    outliers_Iso_Forest = np.where(np.array(labels_Iso_Forest) == -1)
    diabetes_Iso_Forest = diabetes.data.drop(outliers_Iso_Forest[0])
    # remove outliers using LocalOutlierFactor
    LOF = sklearn.neighbors.LocalOutlierFactor(contamination=0.3)
    labels_LOF = LOF.fit_predict(diabetes.data)
    outliers_LOF = np.where(np.array(labels_LOF) == -1)
    diabetes_LOF = diabetes.data.drop(outliers_LOF[0])

    plt.scatter(diabetes_Iso_Forest['plas'], diabetes_Iso_Forest['mass'])
    plt.scatter(diabetes_LOF['plas'], diabetes_LOF['mass'])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Pima Indians Diabetes Database')
    plt.xlabel('plas')
    plt.ylabel('mass')
    plt.show()

def task6():
    diabetes = fetch_openml("diabetes", version=1, as_frame=True)
    # remove outliers using IsolationForest
    x = diabetes.data['plas']
    y = diabetes.data['mass']
    xx, yy = np.meshgrid(x, y)
    Z = np.c_[xx.ravel(), yy.ravel()]
    Iso_Forest = sklearn.ensemble.IsolationForest(contamination=0.3)
    Iso_Forest.fit(Z)
    labels_Iso_Forest = Iso_Forest.predict(Z)
    Z = labels_Iso_Forest.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    plt.scatter(x, y)
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Pima Indians Diabetes Database')
    plt.xlabel('plas')
    plt.ylabel('mass')
    plt.show()

def task7_and_8():
    iris = datasets.load_iris(as_frame=True)
    # divide the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1, stratify=iris.target)
    parameters = {
        'kernel': ('linear', 'rbf'),
        'C': [1, 10],
        'gamma': [0.001, 0.0001]
    }
    clf = model_selection.GridSearchCV(svm.SVC(), parameters, cv=10)
    clf.fit(X_train, y_train)

    # save the best model to a file
    best_model = clf.best_estimator_
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    pvt = pd.pivot_table(
        pd.DataFrame(clf.cv_results_),
        values='mean_test_score',
        index='param_kernel',
        columns='param_C'
    )
    ax = sns.heatmap(pvt)
    plt.show()

    # use RandomizedSearchCV
    clf = model_selection.RandomizedSearchCV(svm.SVC(), parameters, cv=10)
    clf.fit(X_train, y_train)

    pvt = pd.pivot_table(
        pd.DataFrame(clf.cv_results_),
        values='mean_test_score',
        index='param_kernel',
        columns='param_C'
    )
    ax = sns.heatmap(pvt)
    plt.show()

def task8():
    iris = datasets.load_iris(as_frame=True)
    # divide the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1, stratify=iris.target)

    # load the best model from a file
    with open('best_model.pkl', 'rb') as f:
        clf = pickle.load(f)

    predicted = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    print(accuracy)
def task9():
    diabetes = fetch_openml("diabetes", version=1, as_frame=True)
    # remove outliers using IsolationForest
    Iso_Forest = sklearn.ensemble.IsolationForest(contamination=0.3)
    Iso_Forest.fit(diabetes.data)
    labels_Iso_Forest = Iso_Forest.predict(diabetes.data)
    outliers_Iso_Forest = np.where(np.array(labels_Iso_Forest) == -1)
    diabetes_Iso_Forest = diabetes.data.drop(outliers_Iso_Forest[0])
    diabetes.target = diabetes.target.drop(outliers_Iso_Forest[0])
    diabetes.target = np.where(diabetes.target == 'tested_positive', 1, -1)
    scores = cross_val_score(Iso_Forest, diabetes_Iso_Forest, diabetes.target, cv=10, scoring='accuracy')
    print(scores)

def task10_and_11():
    diabetes = fetch_openml("diabetes", version=1, as_frame=True)
    # remove outliers using IsolationForest
    Iso_Forest = sklearn.ensemble.IsolationForest(contamination=0.3)
    Iso_Forest.fit(diabetes.data)
    labels_Iso_Forest = Iso_Forest.predict(diabetes.data)
    outliers_Iso_Forest = np.where(np.array(labels_Iso_Forest) == -1)
    diabetes_Iso_Forest = diabetes.data.drop(outliers_Iso_Forest[0])
    diabetes.target = diabetes.target.drop(outliers_Iso_Forest[0])
    diabetes.target = np.where(diabetes.target == 'tested_positive', 1, -1)
    scores = cross_val_score(Iso_Forest, diabetes_Iso_Forest, diabetes.target, cv=10, scoring='accuracy')
    print(scores)

    # create VotingClassifier with ExtraTreesClassifier and RandomForest
    #clf1 = ExtraTreesClassifier()
    clf1 = LogisticRegression(max_iter=1000, random_state=123)
    clf2 = RandomForestClassifier()
    clf3 = GaussianNB()
    eclf = sklearn.ensemble.VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gn', clf3)], voting='soft')
    sclf = StackingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gn', clf3)], final_estimator=LogisticRegression(max_iter=1000, random_state=123))
    # predict class probabilities for all classifiers
    probas = [c.fit(diabetes_Iso_Forest, diabetes.target).predict_proba(diabetes_Iso_Forest) for c in (clf1, clf2, clf3, eclf, sclf)]
    #print(scores)
    print(np.array(probas)[:, 0])

    # get class probabilities for the first sample in the dataset
    class1_1 = [pr[0, 0] for pr in probas]
    class2_1 = [pr[0, 1] for pr in probas]

    # plotting

    N = 5  # number of groups
    ind = np.arange(N)  # group positions
    width = 0.35  # bar width

    fig, ax = plt.subplots()

    # bars for classifier 1-3
    p1 = ax.bar(ind, np.hstack((class1_1[:-2], np.zeros(2))), width, color="green", edgecolor="k")
    p2 = ax.bar(
        ind + width,
        np.hstack(([class2_1[:-2], np.zeros(2)])),
        width,
        color="lightgreen",
        edgecolor="k",
    )

    # bars for VotingClassifier
    p3 = ax.bar(ind, [0, 0, 0] + list(class1_1[-2:]), width, color="blue", edgecolor="k")
    p4 = ax.bar(ind + width, [0, 0, 0] + list(class2_1[-2:]), width, color="steelblue", edgecolor="k")

    # plot annotations
    plt.axvline(2.8, color="k", linestyle="dashed")
    ax.set_xticks(ind + width)
    ax.set_xticklabels(
        [
            "LogisticRegression\nweight 1",
            "GaussianNB\nweight 1",
            "RandomForestClassifier\nweight 1",
            "VotingClassifier\n(average probabilities)",
            "StackingClassifier\n",
        ],
        rotation=40,
        ha="right",
    )
    plt.ylim([0, 1])
    plt.title("Class probabilities for sample 1 by different classifiers")
    plt.legend([p1[0], p2[0]], ["class 1", "class 2"], loc="upper left")
    plt.tight_layout()
    plt.show()

def task10_1():
    diabetes = fetch_openml("diabetes", version=1, as_frame=True)
    # remove outliers using IsolationForest
    Iso_Forest = sklearn.ensemble.IsolationForest(contamination=0.3)
    Iso_Forest.fit(diabetes.data)
    labels_Iso_Forest = Iso_Forest.predict(diabetes.data)
    outliers_Iso_Forest = np.where(np.array(labels_Iso_Forest) == -1)
    diabetes_Iso_Forest = diabetes.data.drop(outliers_Iso_Forest[0])
    diabetes.target = diabetes.target.drop(outliers_Iso_Forest[0])
    diabetes.target = np.where(diabetes.target == 'tested_positive', 1, -1)

    # create VotingClassifier with ExtraTreesClassifier and RandomForest
    clf1 = LogisticRegression(max_iter=1000, random_state=123)
    clf2 = RandomForestClassifier()
    clf3 = GaussianNB()
    eclf = sklearn.ensemble.VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gn', clf3)], voting='soft')
    sclf = StackingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gn', clf3)], final_estimator=LogisticRegression(max_iter=1000, random_state=123))

    # divide the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(diabetes_Iso_Forest, diabetes.target, test_size=0.2, random_state=1, stratify=diabetes.target)

    accuracy_list = []
    best_accuracy = -1
    best_classifier = None
    classifiers = [clf1, clf2, clf3, eclf, sclf]

    for classifier in classifiers:
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        score = accuracy_score(y_test, predictions)
        accuracy_list.append(score)

        if score > best_accuracy:
            best_accuracy = score
            best_classifier = classifier

    print(accuracy_list)
    with open('best_model_task10.pkl', 'wb') as f:
        pickle.dump(best_classifier, f)

def task12():
    diabetes = fetch_openml("diabetes", version=1, as_frame=True)
    # remove outliers using IsolationForest
    Iso_Forest = sklearn.ensemble.IsolationForest(contamination=0.3)
    Iso_Forest.fit(diabetes.data)
    labels_Iso_Forest = Iso_Forest.predict(diabetes.data)
    outliers_Iso_Forest = np.where(np.array(labels_Iso_Forest) == -1)
    diabetes_Iso_Forest = diabetes.data.drop(outliers_Iso_Forest[0])
    diabetes.target = diabetes.target.drop(outliers_Iso_Forest[0])
    diabetes.target = np.where(diabetes.target == 'tested_positive', 1, -1)

    # divide the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(diabetes_Iso_Forest, diabetes.target, test_size=0.2, random_state=1, stratify=diabetes.target)

    #create random forrest classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    # Get feature importances
    importances = clf.feature_importances_

    # Plot them
    plt.figure(figsize=(10, 5))
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances, color="r", align="center")
    plt.xticks(range(X_train.shape[1]), X_train.columns, rotation='vertical')
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

    # Create a TreeExplainer
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)

def task13():
    diabetes = fetch_openml("diabetes", version=1, as_frame=True)
    # remove outliers using IsolationForest
    Iso_Forest = sklearn.ensemble.IsolationForest(contamination=0.3)
    Iso_Forest.fit(diabetes.data)
    labels_Iso_Forest = Iso_Forest.predict(diabetes.data)
    outliers_Iso_Forest = np.where(np.array(labels_Iso_Forest) == -1)
    diabetes_Iso_Forest = diabetes.data.drop(outliers_Iso_Forest[0])
    diabetes.target = diabetes.target.drop(outliers_Iso_Forest[0])
    diabetes.target = np.where(diabetes.target == 'tested_positive', 1, 0)

    # divide the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(diabetes_Iso_Forest, diabetes.target, test_size=0.2, random_state=1, stratify=diabetes.target)

    # Assume X_train and y_train are your training data and labels
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    model.predict(X_test)
    score = model.score(X_test, y_test)
    print(score)

    # Plot feature importance
    plot_importance(model)
    plt.show()

if __name__ == '__main__':
    #task1()
    #task2()
    #task3()
    #task4()
    #task5()
    #task6()
    #task7_and_8()
    #task8()
    #task9()
    #task10_and_11()
    #task10_1()
    #task12()
    task13()