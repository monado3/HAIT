import pandas as pd
df = pd.read_excel('n_coffee+vs+temp.xlsx')

df.head()

df.loc[:, ['MAX_TEMP']].head()
df.loc[:, 'MAX_TEMP'].head()

import numpy as np
X = np.array(df.loc[:, ['MAX_TEMP']])
X[:5]
y = np.array(df.loc[:, ['N_COFFEE']])
y[:5]

# In[]
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.title('n_coffee vs max_temp')
plt.xlabel('max_temp')
plt.ylabel('n_coffee')
plt.show()

# In[]
df.corr()

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

from sklearn.model_selection import learning_curve
lc = learning_curve(lr, X, y)
lc
lr.fit(X, y)
lr.intercept_
lr.coef_

# In[]
plt.scatter(X, y)
plt.plot(X, lr.predict(X), color = 'red')
plt.title('n_coffee vs temperture')
plt.xlabel('temperture')
plt.ylabel('n_coffee')
plt.show()

# In[]
new_temp = 30
n_pred = lr.predict(new_temp)
print(n_pred)
