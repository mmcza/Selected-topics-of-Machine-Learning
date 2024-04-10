import sklearn
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

from sklearn.model_selection import train_test_split
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt

def main():
    titanic = fetch_openml("Titanic", version=1, as_frame=True)
    #print(titanic.data)
    print(titanic.data.describe())
    # remove column home.dest
    titanic.data.drop(columns=['home.dest'])
    titanic.data.drop(columns=['boat'])
    titanic.data.drop(columns=['body'])
    # group values in age column by each 5 years
    bins = list(np.arange(0, 180, 10))
    titanic.data['age_group'] = pd.cut(titanic.data['age'], bins=bins, labels=bins[:-1])
    print(titanic.data)
    # divide the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(titanic['data'], titanic['target'], test_size=0.1, random_state=42)
    # check distribution of the dataset in training and testing
    # print(y_train.value_counts())
    # create random guess for x_test
    y_test_guess = np.random.choice([0, 1], p=[0.625, 0.375], size=X_test.shape[0])
    #print(y_test_guess)
    # calculate accuracy
    accuracy = (np.int_(y_test.to_numpy()) == y_test_guess).mean()
    print('Random guess accuracy: ', accuracy)
    # X_train_copy = X_train.copy()
    # msno.matrix(X_train_copy)
    # print(X_train_copy)
    # impute missing values for age by using data from sibsp and parch columns and IterativeImputer
    imputer = IterativeImputer(max_iter=10, random_state=42, missing_values=np.nan, sample_posterior=True, min_value=0, max_value=100)
    X_train_imputed = imputer.fit_transform(X_train[['sibsp', 'parch', 'age_group']])
    X_train_imputed = pd.DataFrame(X_train_imputed, columns=['sibsp', 'parch', 'age_group'])
    X_train_imputed['age_group'] = pd.cut(X_train_imputed['age_group'], bins=bins, labels=bins[:-1])
    print(X_train_imputed.describe())








if __name__ == "__main__":
    main()