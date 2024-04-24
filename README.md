# Selected issues of Machine Learning - laboratory

## List of exercises
| Laboratory #     | Topic                                                                                                                                           |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| [Lab #1](/Lab1/) | Loading data, splitting data into training and testing sets, Decision Tree Classifier                                                           |
| [Lab #2](/Lab2/) | Pipelines, Scallers, SVM, metrics: accuracy, confusion matrix, mse; LinearRegression, DecisionTreeRegression                                    |
| [Lab #3](/Lab3/) | Scallers, data visualization, DecisionTreeClassifier, pandas, DecisionBoundaryDisplay                                                           |
| [Lab #4](/Lab4/) | Imputting missing data, Removing outliers, Creating ProfileReport, Cross validation, RandomizedSearchCV, Voting Classifier, Stacking Classifier |
| [Lab #5](/Lab5/) | Prediction of Titanic survivors, Feature engineering, Binary Classification, Classifiers comparison                                             |
| [Lab #6](/Lab6/) | MLflow *, MLOps, Saving model, parameters and statistics; autologging                                                                           |

\* to run [Lab #6](/Lab6/) you need to have mlflow installed and run a command:
```
mlflow server --host 127.0.0.1 --port 8080  
```
in a result when you go in your browser to `http://127.0.0.1:8080/` you should be able to see the following:

![MLflow](/Lab6/screenshot/wzum_lab6_mlflow.png)
