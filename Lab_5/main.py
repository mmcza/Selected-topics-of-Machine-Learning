import sklearn
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # import the data
    titanic = fetch_openml("Titanic", version=1, as_frame=True)
    pd.set_option('display.max_columns', None)
    #print(titanic.data.describe())

    # remove column home.dest, boat and body
    titanic.data = titanic.data.drop(columns=['home.dest', 'boat', 'body'], inplace=False, errors='ignore')

    #print(titanic.data.head())

    # print datatypes and number of missing values in each column
    #print(titanic.data.dtypes)
    print(titanic.data.isnull().sum())

    # group values in age column by each 10 years
    bins = list(np.arange(0, 90, 10))
    titanic.data['age_group'] = pd.cut(titanic.data['age'], bins=bins, labels=bins[:-1])

    # get title from name column
    titanic.data['title'] = titanic.data['name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    #print(titanic.data['title'].value_counts())
    title_category = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master", "Dr": "Dr", "Rev": "Rev",
                      "Col": "Military", "Mlle": "Miss", "Major": "Military", "Ms": "Miss", "Mme": "Mrs", "Sir": "Nobility",
                      "Capt": "Military", "Lady": "Nobility", "the Countess": "Nobility", "Jonkheer": "Nobility",
                      "Don": "Nobility", "Dona": "Nobility"}
    titanic.data['title_categorical'] = titanic.data['title'].map(title_category)
    #print(titanic.data['title_categorical'].value_counts())

    # sum of sibsp and parch columns
    titanic.data['family_size'] = titanic.data['sibsp'] + titanic.data['parch'] + 1
    #print(titanic.data['family_size'].value_counts())
    titanic.data['fare_per_person'] = titanic.data['fare'] / titanic.data['family_size']
    print(titanic.data['fare_per_person'].describe())


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

    # check if someone was travelling alone
    X_train['is_alone'] = (X_train['family_size'] == 1).astype(int)
    X_test['is_alone'] = (X_test['family_size'] == 1).astype(int)

    # impute missing values for fare and fare_per_person by using data from sibsp, parch and pclass columns and KNNImputer
    imputer = KNNImputer(n_neighbors=2, missing_values=np.nan)
    X_train[['fare_per_person', 'sibsp', 'parch', 'pclass']] = imputer.fit_transform(X_train[['fare_per_person', 'sibsp', 'parch', 'pclass']])
    X_test[['fare_per_person', 'sibsp', 'parch', 'pclass']] = imputer.transform(X_test[['fare_per_person', 'sibsp', 'parch', 'pclass']])
    # search for missing values in column fare and replace it with fare_per_person * family_size
    X_train['fare'] = X_train['fare'].fillna(X_train['fare_per_person'] * X_train['family_size'])
    X_test['fare'] = X_test['fare'].fillna(X_test['fare_per_person'] * X_test['family_size'])

    # get cabin letter from cabin column if it is not missing
    X_train['deck'] = X_train['cabin'].str[0]
    X_test['deck'] = X_test['cabin'].str[0]
    # if deck is equal T then replace it with A - only one person was in deck T
    X_train['deck'] = X_train['deck'].replace('T', 'A')
    X_test['deck'] = X_test['deck'].replace('T', 'A')

    # use LabelEncoder to encode the deck column
    deck_le = LabelEncoder()
    X_train['deck_encoded'] = deck_le.fit_transform(X_train['deck'])
    X_test['deck_encoded'] = deck_le.transform(X_test['deck'])
    #print(X_train['deck_encoded'].value_counts())

    # use LabelEncoder to encode the age_group column
    age_group_le = LabelEncoder()
    X_train['age_group_encoded'] = age_group_le.fit_transform(X_train['age_group'])
    X_test['age_group_encoded'] = age_group_le.transform(X_test['age_group'])
    #print(X_train['age_group_encoded'].value_counts())

    # use LabelEncoder to encode the title_categorical column
    title_categorical_le = LabelEncoder()
    X_train['title_categorical_encoded'] = title_categorical_le.fit_transform(X_train['title_categorical'])
    X_test['title_categorical_encoded'] = title_categorical_le.transform(X_test['title_categorical'])
    #print(X_train['title_categorical_encoded'].value_counts())

    # use LabelEncoder to encode the sex column
    sex_le = LabelEncoder()
    X_train['sex_encoded'] = sex_le.fit_transform(X_train['sex'])
    X_test['sex_encoded'] = sex_le.transform(X_test['sex'])
    # print(X_train['sex_encoded'].value_counts())

    # use LabelEncoder to encode the embarked column
    embarked_le = LabelEncoder()
    X_train['embarked_encoded'] = embarked_le.fit_transform(X_train['embarked'])
    X_test['embarked_encoded'] = embarked_le.transform(X_test['embarked'])
    # print(X_train['embarked_encoded'].value_counts())

    # find correlation between columns
    corr = X_train[['deck_encoded', 'pclass', 'fare_per_person', 'family_size', 'title_categorical_encoded', 'sex_encoded',
                    'embarked_encoded', 'sibsp', 'parch', 'age_group_encoded', 'is_alone']].corr()

    # find correlation between columns where deck is not missing
    corr_no_missing = X_train[X_train['deck_encoded'] != 7][['deck_encoded', 'pclass', 'fare_per_person', 'family_size',
                                        'title_categorical_encoded', 'sex_encoded', 'embarked_encoded', 'sibsp',
                                        'parch', 'age_group_encoded', 'is_alone']].corr()

    # replacing missing values in deck column with -1
    X_train['deck_encoded_no_missing'] = X_train['deck_encoded'].replace(7, -1)
    X_test['deck_encoded_no_missing'] = X_test['deck_encoded'].replace(7, -1)

    # impute missing values for deck column by using data from pclass and fare_per_person columns and IterativeImputer
    deck_imputer = KNNImputer(n_neighbors=2, missing_values=-1)
    X_train[['deck_encoded_no_missing', 'pclass', 'fare_per_person', 'embarked_encoded', 'is_alone']] = (
        deck_imputer.fit_transform(X_train[['deck_encoded_no_missing', 'pclass', 'fare_per_person',
                                            'embarked_encoded', 'is_alone']]))
    X_test[['deck_encoded_no_missing', 'pclass', 'fare_per_person', 'embarked_encoded', 'is_alone']] = (
        deck_imputer.transform(X_test[['deck_encoded_no_missing', 'pclass', 'fare_per_person',
                                       'embarked_encoded', 'is_alone']]))


    # check correlation after imputing missing values
    corr_imputed = X_train[['deck_encoded_no_missing', 'pclass', 'fare_per_person', 'family_size', 'title_categorical_encoded', 'sex_encoded',
                    'embarked_encoded', 'sibsp', 'parch', 'age_group_encoded', 'is_alone']].corr()

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
    # Plot correlation matrix 1
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0])
    axes[0].set_title('Initial correlation Matrix')

    # Plot correlation matrix 2
    sns.heatmap(corr_no_missing, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1])
    axes[1].set_title('Correlation Matrix without missing values for deck column')

    # Plot correlation matrix 3
    sns.heatmap(corr_imputed, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[2])
    axes[2].set_title('Correlation Matrix after imputing missing values for deck column')
    plt.show()

    # replace missing values in age_group_encoded column with -1
    X_train['age_group_encoded'] = X_train['age_group_encoded'].replace(8, -1)
    X_test['age_group_encoded'] = X_test['age_group_encoded'].replace(8, -1)

    # impute missing values for age_group_encoded column by using IterativeImputer
    age_group_imputer = IterativeImputer(max_iter=10, random_state=42, missing_values=-1)
    X_train[['age_group_encoded', 'family_size', 'title_categorical_encoded', 'is_alone', 'parch']] = (
        age_group_imputer.fit_transform(X_train[['age_group_encoded', 'family_size', 'title_categorical_encoded', 'is_alone', 'parch']]))
    X_test[['age_group_encoded', 'family_size', 'title_categorical_encoded', 'is_alone', 'parch']] = (age_group_imputer.transform(
        X_test[['age_group_encoded', 'family_size', 'title_categorical_encoded', 'is_alone', 'parch']]))

    # check correlation after imputing missing values
    corr_no_missing_age = X_train[pd.notna(X_train['age'])][['deck_encoded', 'pclass', 'fare_per_person', 'family_size',
                                        'title_categorical_encoded', 'sex_encoded', 'embarked_encoded', 'sibsp',
                                        'parch', 'age_group_encoded', 'is_alone']].corr()

    corr_imputed_age = X_train[['deck_encoded', 'pclass', 'fare_per_person', 'family_size',
                                        'title_categorical_encoded', 'sex_encoded', 'embarked_encoded', 'sibsp',
                                        'parch', 'age_group_encoded', 'is_alone']].corr()

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
    # Plot correlation matrix 1
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0])
    axes[0].set_title('Initial correlation Matrix')
    #Plot correlation matrix 2
    sns.heatmap(corr_no_missing_age, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1])
    axes[1].set_title('Correlation Matrix without missing values for age_group_encoded column')
    #Plot correlation matrix 3
    sns.heatmap(corr_imputed_age, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[2])
    axes[2].set_title('Correlation Matrix after imputing missing values for age_group_encoded column')
    plt.show()

    # impute missing values for embarked_encoded column by using IterativeImputer
    embarked_imputer = IterativeImputer(max_iter=10, random_state=42, missing_values=-1)
    X_train[['embarked_encoded', 'fare_per_person', 'deck_encoded', 'pclass']] = (
        embarked_imputer.fit_transform(X_train[['embarked_encoded', 'fare_per_person', 'deck_encoded', 'pclass']]))
    X_test[['embarked_encoded', 'fare_per_person', 'deck_encoded', 'pclass']] = (
        embarked_imputer.transform(X_test[['embarked_encoded', 'fare_per_person', 'deck_encoded', 'pclass']]))


    # check final correlation
    data_train = pd.concat([X_train, y_train], axis=1)
    #print(data_train.columns)
    correlation_matrix = data_train[['survived', 'pclass', 'sibsp', 'parch',
       'family_size', 'fare_per_person', 'is_alone', 'deck_encoded',
       'age_group_encoded', 'title_categorical_encoded', 'sex_encoded',
       'embarked_encoded', 'deck_encoded_no_missing']].corr()

    # Create plot
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Final correlation Matrix')
    plt.show()

    # create Random Forest Classifier
    rfc_clf = RandomForestClassifier(random_state=42)
    rfc_clf.fit(X_train[['pclass', 'fare_per_person', 'is_alone', 'deck_encoded_no_missing',
       'age_group_encoded', 'title_categorical_encoded', 'sex_encoded',
       'embarked_encoded']], y_train)
    rfc_prediction = rfc_clf.predict(X_test[['pclass', 'fare_per_person', 'is_alone', 'deck_encoded_no_missing',
       'age_group_encoded', 'title_categorical_encoded', 'sex_encoded',
       'embarked_encoded']])

    # calculate accuracy
    accuracy_rfc = accuracy_score(y_test, rfc_prediction)
    print("Accuracy of prediction for Random Forest Classifier is: "+str(accuracy_rfc))

if __name__ == "__main__":
    main()