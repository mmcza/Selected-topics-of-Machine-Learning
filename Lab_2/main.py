import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
def task1():
    digits = datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits['data'], digits['target'], test_size=0.3, random_state=1)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    # print(predicted)
    # print(y_test)
    accuracy = accuracy_score(y_test, predicted)
    print(accuracy)

def task2_1():
    wine_pandas = datasets.load_wine(as_frame=True)
    wine = datasets.load_wine()

    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(wine['data'], wine['target'], test_size=0.2, random_state=1)
    clf_w = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf_w.fit(X_train_w, y_train_w)
    predicted_w = clf_w.predict(X_test_w)
    # print(predicted)
    # print(y_test)
    accuracy_w = accuracy_score(y_test_w, predicted_w)
    print("Accuracy of prediction for Wine database is: "+str(accuracy_w))

def task2_2():
    X_mc, y_mc = datasets.make_classification(random_state=42)
    X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(X_mc, y_mc, test_size=0.2, random_state=1)
    clf_mc = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf_mc.fit(X_train_mc, y_train_mc)
    predicted_mc = clf_mc.predict(X_test_mc)
    # print(predicted)
    # print(y_test)
    accuracy_mc = accuracy_score(y_test_mc, predicted_mc)
    print("Accuracy of prediction for 'Make classification' is: "+str(accuracy_mc))

def task4():
    digits = datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits['data'], digits['target'], test_size=0.3, random_state=1)
    clf_d = DecisionTreeClassifier()
    clf_d.fit(X_train, y_train)
    predicted_d = clf_d.predict(X_test)
    c_matr = confusion_matrix(y_test, predicted_d)
    print(c_matr)
    c_rep = clf_d, classification_report(y_test, predicted_d)
    print(c_rep)

def task5():
    battery_training_data = pd.read_csv('trainingdata.txt', header=None)
    battery_training_data.columns = ['charged', 'lasted']
    # print(battery_training_data)
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(battery_training_data[['charged']],
                                                                battery_training_data[['lasted']], test_size=0.2,
                                                                random_state=1)

    # Basic Linear Regression
    reg_basic = LinearRegression()
    reg_basic.fit(X_train_b, y_train_b)
    y_pred_basic = reg_basic.predict(X_test_b)

    mse_basic = mean_squared_error(y_test_b, y_pred_basic)
    mae_basic = mean_absolute_error(y_test_b, y_pred_basic)
    r2_basic = r2_score(y_test_b, y_pred_basic)
    print("Mean square error of prediction for Linear Regression: "+str(mse_basic*60)+" minutes")
    print("Mean absolute error of prediction for Linear Regression: "+str(mae_basic*60)+" minutes")
    print("R2 score for Linear Regression: "+str(r2_basic))
    print("\n")

    # Decision Tree Regressor
    reg_tree = DecisionTreeRegressor()
    reg_tree.fit(X_train_b, y_train_b)
    y_pred_tree = reg_tree.predict(X_test_b)

    mse_tree = mean_squared_error(y_test_b, y_pred_tree)
    mae_tree = mean_absolute_error(y_test_b, y_pred_tree)
    r2_tree = r2_score(y_test_b, y_pred_tree)
    print("Mean square error of prediction for Decision Tree Regression: "+str(mse_tree*60)+" minutes")
    print("Mean absolute error of prediction for Decision Tree Regression: "+str(mae_tree*60)+" minutes")
    print("R2 score for Decision Tree Regression: "+str(r2_tree))



    plt.plot(X_test_b, y_test_b, 'bo', label='True values')
    plt.plot(X_test_b, y_pred_basic, 'go', label='Predicted values - Linear Regression')
    plt.plot(X_test_b, y_pred_tree, 'r*', label='Predicted values - Decision Tree Regression')
    plt.xlabel("Charging time [h]")
    plt.ylabel("Lasted time [h]")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #task1()
    #task2_1()
    #task2_2()
    #task4()
    task5()