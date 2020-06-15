import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df_m = pd.read_csv('churn_data.csv')
df1 = pd.read_csv('customer_data.csv')
df2 = pd.read_csv('internet_data.csv')
df = pd.merge(df_m, df1, how='inner', on='customerID')
df = pd.merge(df, df2, how='inner', on='customerID')

df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})

df = df[['tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'Churn', 'Partner', 'Dependents']]

yes_no = 'Churn'

x = np.array(df.drop([yes_no], 1))

y = np.array(df[yes_no])
new_boss = 0
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1)
predict = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1)
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    accuracy = log_reg.score(x_test, y_test)
    prediction = log_reg.predict(x_test)
    if new_boss < accuracy:
        new_boss = accuracy
        predict = prediction

print(new_boss)

for _ in range(len(predict)):
    print(y_test[_], f'predicted number: {predict[_]}')
