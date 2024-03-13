import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


#Link to list of exercises https://colab.research.google.com/drive/1XwQR4uX9uB1XiLYm1j4pxaET37A1B7JJ#scrollTo=1qN1KIJV7eKh

### TODO 2

digits = datasets.load_digits()

# np.unique(digits.images)
# for i in range(5):
#     plt.gray()
#     plt.matshow(digits.images[i])
#     plt.show()

# print(digits['DESCR'])

# print(len(digits.images))

### TODO 3

X_train, X_test, y_train, y_test = train_test_split(digits['data'], digits['target'], test_size=0.3, random_state=1)

### TODO 4

olivetti_faces = datasets.fetch_olivetti_faces()
# print(olivetti_faces['DESCR'])
X_train_of, X_test_of, y_train_of, y_test_of = train_test_split(olivetti_faces['data'], olivetti_faces['target'], test_size=0.2, random_state=1)
for i in np.unique(olivetti_faces.target):
    print(str(i)+" - "+str(np.count_nonzero(olivetti_faces.target == i)))

X_train_messed_up = olivetti_faces.data[0:320]
X_test_messed_up = olivetti_faces.data[320:400]
y_train_messed_up = olivetti_faces.target[0:320]
y_test_messed_up = olivetti_faces.target[320:400]

### Todo 5

wine_pandas = datasets.load_wine(as_frame = True)
# print(wine_pandas.data.describe())
# print(wine_pandas.data.columns)
# print(wine_pandas.data)
wine = datasets.load_wine()
# print(wine.feature_names)

X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(wine['data'], wine['target'], test_size=0.2, random_state=1)

### Todo 6

X_mc, y_mc = datasets.make_classification(random_state=42)
# plt.figure()
# plt.plot(X_mc, y_mc)
# print(X_mc)
# plt.show()

### Todo 7

tic_tac_toe = datasets.fetch_openml('tic-tac-toe')
# print(tic_tac_toe.DESCR)
# print(tic_tac_toe.data)

### Todo 8

battery_training_data = pd.read_csv('trainingdata.txt', header= None)
battery_training_data.columns = ['charged', 'lasted']
# print(battery_training_data)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(battery_training_data['charged'], battery_training_data['lasted'], test_size=0.2, random_state=1)

### Hello World

X_and = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
y_and = [0, 0, 0, 1]

clf_and = DecisionTreeClassifier()
clf_and.fit(X_and, y_and)

print(clf_and.predict([[0, 1]]))

### Todo 9

X_or = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
y_or = [0, 1, 1, 1]

clf_or = DecisionTreeClassifier()
clf_or.fit(X_or, y_or)

print(clf_or.predict([[0, 1]]))

### Todo 10

plt.figure()
sklearn.tree.plot_tree(clf_and, filled= True)
plt.title("Decision tree for AND gate")
plt.show()

plt.figure()
sklearn.tree.plot_tree(clf_or, filled= True)
plt.title("Decision tree for OR gate")
plt.show()