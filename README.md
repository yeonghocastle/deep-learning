# deep-learning
import numpy as np 
import matplotlib.pyplot as plt

rs = np.random.RandomState(10)
x = 10 * rs.rand(100)
y = 3 * x + 2 * rs.rand(100)
plt.scatter(x,y,s=10)


from sklearn.linear_model import LinearRegression
regr = LinearRegression()


X = x.reshape(-1, 1)
X.shape, y.shape

regr.fit(X,y)
regr.coef_
regr.intercept_

x_new = np.linspace(-1, 11, num=100)
X_new = x_new.reshape(-1,1)
X_new.shape

y_pred = regr.predict(X_new)
plt.plot(x_new, y_pred, c="red")
plt.scatter(x, y, s=10)

from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y,y_pred))
