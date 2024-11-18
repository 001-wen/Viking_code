import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(1)
x=10*np.random.rand(500,1)
y=5*x+10*np.random.randn(500,1)
model=LinearRegression()
model.fit(x,y)
x_pred=np.array([[0],[10]])
y_pred=model.predict(x_pred)
fig=plt.figure()
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(x,y,color="green",label="Points")
plt.plot(x_pred,y_pred,c="red",label="Regression Line")
plt.title("Linear Regression")
plt.legend()
plt.show()