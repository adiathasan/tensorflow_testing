import matplotlib
import sklearn
from docutils.nodes import inline
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

digits = load_digits()

print(digits.data.shape)

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=.20, random_state=2)

logic_reg = LogisticRegression()

logic_reg.fit(x_train, y_train)

accuracy = logic_reg.score(x_test, y_test)

prediction = logic_reg.predict(x_test)

for _ in range(len(prediction)):
    print(x_test[_], y_test[_], f'predicted number: {prediction[_]}')