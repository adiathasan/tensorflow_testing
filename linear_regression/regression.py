import pandas as pd
import numpy as np
import sklearn
from pandas.plotting import scatter_matrix
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pp
import pickle
from matplotlib import style
import quandl

data = quandl.get('LPPM/PALL')

df = data[['USD AM', 'GBP AM', 'USD PM', 'GBP PM']]

df.dropna(inplace=True)

to_predict = 'GBP PM'

x = np.array(df.drop([to_predict], 1))

y = np.array(df[to_predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.1)

highest_accuracy = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)

    accuracy = linear.score(x_test, y_test)

    if accuracy > highest_accuracy:
        highest_accuracy = accuracy

        with open('save-analysis.pickle', 'wb') as f:
            pickle.dump(linear, f)

super_power = open('save-analysis.pickle', 'rb')

linear = pickle.load(super_power)


prediction = linear.predict(x_test)


for _ in range(len(prediction)):
    print(x_test[_], y_test[_], f'prediction: {prediction[_]}')
