import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

b=0
m=0
learning_rate=0.01
num_iterations=5
x=np.array([6.4982,5.5317,8.7134,7.0991])
y=np.array([6.2467,9.1435,12.6148,11.6657])

for i in range(num_iterations):
    yp = m*x+b
    Dm = (-1/len(x))*sum(x*(y-yp))
    Db = (-1/len(x))*sum(y-yp)
    m = m-learning_rate*Dm
    b = b-learning_rate*Db
    J = 0.5/len(x)*sum(np.power(m*x+b-y,2))
    print("Iteration: {}, m: {}, b: {}, J: {}".format(i,m,b,J))