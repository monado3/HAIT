import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

a = np.array([1, 2, 2, 4, 5, 5, 6, 7, 7, 7])
np.mean(a)
np.median(a)

d = np.array([1, 2, 2, 3, 4, 4, 4, 5, 5, 6, 7])
stats.mode(d)

sum((a - (np.ones(len(a)) * np.mean(a)))**2)

np.var(a)
np.std(a)

df = pd.DataFrame(a)
df.describe()

# In[]:
eng = np.random.normal(70,10,1000)
plt.hist(eng)
plt.show()

# In[]:
#スタージェスの公式
import math
n_bins = math.log(1000,2) + 1
print(n_bins)

plt.hist(eng,bins=11)
plt.show()

# In[]:
test = math.log(10000,2) + 1
test
