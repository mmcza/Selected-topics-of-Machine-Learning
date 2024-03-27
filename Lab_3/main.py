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
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from mlxtend.plotting import plot_decision_regions

def task1():
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

def task2():
    # import iris dataset as dataframe
    iris = datasets.load_iris(as_frame=True)
    # use pandas to describe the dataset
    print(iris.data.describe())
    # TASK 3
    # divide the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.2, random_state=1)
    # TASK 4
    # check distribution of the dataset in training and testing
    print(iris['target'].value_counts())
    print(y_train.value_counts())

    # divide the dataset into stratified training and testing
    X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.2, random_state=1, stratify=iris['target'])
    # check distribution of the dataset in training and testing
    print(y_train.value_counts())
    plt.scatter(X_train['sepal length (cm)'], X_train['sepal width (cm)'])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.show()
    # TASK 5
    # preprocess data with MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    #print(scaler.data_max_)
    #print(scaler.transform(X_train))
    X_train_scalled = scaler.transform(X_train)
    plt.scatter(X_train_scalled[:, 0], X_train_scalled[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')

    # TASK 6
    # preprocess data with StandardScaler
    s_scaler = StandardScaler()
    s_scaler.fit(X_train)
    #print(s_scaler.data_max_)
    #print(s_scaler.transform(X_train))
    X_train_s_scalled = s_scaler.transform(X_train)
    plt.scatter(X_train_s_scalled[:, 0], X_train_s_scalled[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.show()

    # TASK 7
    # use pipeline to scale the data
    pipe = make_pipeline(StandardScaler())
    pipe.fit(X_train)
    X_train_pipe = pipe.transform(X_train)
    #print(X_train_pipe)

    # TASK 8
    # train a DecissionTree classifier with data
    clf = DecisionTreeClassifier()
    #print(X_train[['sepal length (cm)', 'sepal width (cm)']])
    #print(X_train[:, [0, 1]])
    clf.fit(X_train[['sepal length (cm)', 'sepal width (cm)']], y_train)
    predicted = clf.predict(X_test[['sepal length (cm)', 'sepal width (cm)']])
    # get accuracy of the prediction
    accuracy = accuracy_score(y_test, predicted)
    print("Accuracy of prediction for 'Iris' is: "+str(accuracy))

    # TASK 9
    # create a plot for the decision tree DecisionBoundaryDisplay
    plt.figure()
    #iris_data_list = iris['data'].values[:, [0, 1]]
    #print(iris_data_list)
    plot_decision_regions(iris['data'][['sepal length (cm)', 'sepal width (cm)']].to_numpy(), iris['target'].to_numpy(), clf=clf, legend=2)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('sepal width [cm]')
    plt.title('DecissionTree on Iris')
    plt.show()

if __name__ == '__main__':
    #task1()
    task2()