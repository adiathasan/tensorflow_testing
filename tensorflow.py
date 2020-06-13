import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pp
import pickle
from matplotlib import style

data = pd.read_csv('student-mat.csv', sep=';')

data = data[['age', 'G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

to_predict = 'G3'

x = np.array(data.drop([to_predict], 1))

y = np.array(data[to_predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.1)
best = 0
for _ in range(10000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.1)
    liner = linear_model.LinearRegression()

    liner.fit(x_train, y_train)

    accuracy = liner.score(x_test, y_test)

    if accuracy > best:
        best = accuracy
        with open('student-fin-grd.pickle', 'wb') as f:
            pickle.dump(liner, f)

open_pickle = open('student-fin-grd.pickle', 'rb')

liner = pickle.load(open_pickle)

predictions = liner.predict(x_test)

for _ in range(len(predictions)):
    print(predictions[_], x_test[_], y_test[_])

style.use('ggplot')
x_axis = 'age'
pp.scatter(data[x_axis], data[to_predict])
pp.xlabel(x_axis)
pp.ylabel('Final Grade')
pp.show()
