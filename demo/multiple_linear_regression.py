from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read and split data into two parts: features (X) and the result (Y)
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# split data into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/3, random_state=0)

# find the best fit line
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)

# visualize
plt.scatter(X_train, Y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color='blue')


# plt.scatter(X_test, Y_test, color="green")
# plt.title('Salary vs Experience (Test set)')
plt.show()
