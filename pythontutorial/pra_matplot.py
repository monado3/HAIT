import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
x_1 = np.random.normal(50, 10, 100)
y_1 = x_1 + np.random.normal(0, 5, 100)

x_1[:24]
y_1[:24]
plt.scatter(x_1, y_1)
plt.show()

# In[]:
plt.scatter(x_1, y_1)
plt.title('The relationship between x and y')
plt.show()

# In[]:
plt.scatter(x_1, y_1)
plt.title('The relationship between x and y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# In[]:
plt.scatter(x_1, y_1, color='y')
plt.title('The relationship between x and y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# In[]:
x_2 = np.random.normal(30, 5, 100)
y_2 = 2.5 * x_2 + np.random.normal(0, 10, len(x_2))
plt.scatter(x_1, y_1, color='red')
plt.scatter(x_2, y_2, color='blue')
plt.title('The relationship between x and y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# In[]:
plt.scatter(x_1,y_1,color='red',marker='s')
plt.scatter(x_2,y_2,color='blue',marker='*')
plt.title('The relationship between x and y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# In[]:
plt.scatter(x_1,y_1,color='red',marker='s',label='data1')
plt.scatter(x_2,y_2,color='blue',marker='*',label='data2')
plt.title('The relationship between x and y')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.show()

# In[]:
plt.scatter(x_1,y_1,color='red',marker='s',label='data1')
plt.scatter(x_2,y_2,color='blue',marker='*',label='data2')
plt.title('The relationship between x and y')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.xticks([20,50,80])
plt.yticks([20,50,80,110])
plt.show()

# In[]:
plt.scatter(x_1,y_1,color='red',marker='s',label='data1')
plt.scatter(x_2,y_2,color='blue',marker='*',label='data2')
plt.title('The relationship between x_1 and y_1')
plt.xlabel('feature')
plt.ylabel('value')
plt.legend(loc='upper right')
plt.xticks(np.arange(10,80,5))
plt.yticks(np.arange(10,130,10))
plt.show()

# In[]:
plt.scatter(x_1,y_1,color='red',marker='s',label='data1')
plt.scatter(x_2,y_2,color='blue',marker='*',label='data2')
plt.title('The relationship between x_1 and y_1')
plt.xlabel('feature')
plt.ylabel('value')
plt.legend(loc='upper right')
plt.xticks(np.arange(10,80,5))
plt.yticks(np.arange(10,130,10))
plt.grid()
plt.show()

# Graph
# In[]:
x_3=np.arange(-10,10,0.1)
y_3=0.001*(x_3**3 + x_3**2 + x_3 + 1)
plt.plot(x_3,y_3)
plt.show()

# In[]:
plt.plot(x_3,y_3, label='cubic')
plt.title('a cubic function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.xticks(np.arange(-10,11,1))
plt.yticks(np.arange(-1,1.5,0.25))
plt.grid()
plt.show()

# In[]:
plt.plot(x_3, y_3, label='cubic')
plt.title('a cubic function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.xticks(np.arange(-10,11,1))
plt.yticks(np.arange(-1,1.5,0.25))
plt.grid()
plt.hlines([0],-3,3,linestyles='dashed',color='gray')
plt.vlines([0],-1,1,linestyles='dashed',color='gray')
plt.xlim([-5,5])
plt.ylim([-1,1])
plt.show()
