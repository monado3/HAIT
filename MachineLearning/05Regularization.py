import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
# In[] LASSO
# In[] RidgeReg.
# In[]
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

# In[]


def cubic(X):
  y = 0.0001 * (X**3 + X**2 + X + 1)
  return y


np.random.seed(3)
X = np.random.normal(0, 10, 30)
y = cubic(X)

y += np.random.normal(0, 0.25, len(y))
X = X.reshape(-1, 1)

X_plot = np.arange(-25, 25, 0.1).reshape(-1, 1)
y_plot = cubic(X_plot)
X_plot

# In[]
plt.scatter(X, y)
plt.plot(X_plot, y_plot, color='gray')
plt.ylim([-1.6, 1.6])
plt.show()

lr = LinearRegression()
pol = PolynomialFeatures(degree=7)
X_pol = pol.fit_transform(X)
lr.fit(X_pol, y)
X_plot_pol = pol.fit_transform(X_plot)
y_plot_pol = lr.predict(X_plot_pol)
X_plot_pol
y_plot_pol
# In[]
plt.scatter(X, y)
plt.plot(X_plot, y_plot, color='gray')
plt.plot(X_plot, y_plot_pol, color='green')
plt.ylim([-1.6, 1.6])
plt.show()

model_ridge = Ridge(alpha=1000)
model_ridge.fit(X_pol, y)

# In[]
plt.scatter(X, y)
plt.plot(X_plot, y_plot, color='gray')
plt.plot(X_plot, model_ridge.predict(X_plot_pol), color='red')
plt.plot(X_plot, y_plot_pol, color='green')
plt.ylim([-1.6, 1.6])
plt.show()

# In[]
lr.coef_
model_ridge.coef_
la.norm(lr.coef_)
la.norm(model_ridge.coef_)

# In[] LASSO
from sklearn.linear_model import Lasso
model_lasso = Lasso(alpha=1000)
model_lasso.fit(X_pol, y)
plt.scatter(X,y)
plt.plot(X_plot, y_plot, color = 'gray')
plt.plot(X_plot, y_plot_pol, color = 'green')
plt.plot(X_plot, model_lasso.predict(X_plot_pol), color = 'red')
plt.ylim([-1.6, 1.6])
plt.show()


# In[]
lr.coef_
model_lasso.coef_
la.norm(lr.coef_, ord = 1)
la.norm(model_lasso.coef_, ord = 1)

# In[] ElasticNet
from sklearn.linear_model import ElasticNet
model_en = ElasticNet(alpha = 1000, l1_ratio=0.9)
model_en.fit(X_pol, y)

# In[]
plt.scatter(X, y)
plt.plot(X_plot, y_plot, color = 'gray')
plt.plot(X_plot, y_plot_pol, color = 'green')
plt.plot(X_plot, model_en.predict(X_plot_pol), color = 'red')
plt.ylim([-1.6, 1.6])
plt.show()

# In[]
lr.coef_
model_en.coef_
la.norm(lr.coef_)
la.norm(model_en.coef_)

la.norm(lr.coef_, ord = 1)
la.norm(model_en.coef_, ord = 1)
