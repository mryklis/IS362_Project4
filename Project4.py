import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

page = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
df1 = pd.read_csv(page, index_col=False, header=None, names=['Poisonous?', 'Cap Color', 'Odor'], usecols=[0,3,5])

# poisonous = 1, edible = 0
df1.replace(to_replace={'Poisonous?':{'p':1, 'e': 0}}, inplace=True)

C = pd.Series(df1['Cap Color'])
f = pd.get_dummies(C)


O = pd.Series(df1['Odor'])
g = pd.get_dummies(O)


new_df = pd.concat([f, g, df1['Poisonous?']], axis=1)
cols = list(new_df.iloc[:, :-1])



X = new_df.iloc[:, :-1].values
y = new_df.iloc[:, 1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

linreg = LinearRegression()
linreg.fit(X_train, y_train)
print(linreg.intercept_)
print(linreg.coef_)

print(list(zip(cols, linreg.coef_)))

y_pred = linreg.predict(X_test)

true = [1, 0]
pred = [1, 0]

print(metrics.mean_absolute_error(true, pred))
print(metrics.mean_squared_error(true, pred))
print(np.sqrt(metrics.mean_squared_error(true, pred)))

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))




# use the list to select a subset of the original DataFrame
X = new_df.iloc[:, 11:-1].values

# select a Series from the DataFrame
y = new_df.iloc[:, 1].values

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)

# make predictions on the testing set
y_pred = linreg.predict(X_test)

# compute the RMSE of our predictions
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))




# use the list to select a subset of the original DataFrame
X = new_df.iloc[:, 1:10].values

# select a Series from the DataFrame
y = new_df.iloc[:, 1].values

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)

# make predictions on the testing set
y_pred = linreg.predict(X_test)

# compute the RMSE of our predictions
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))