import numpy as np
import matplotlib.pyplot as plt

def initialize(dim):
  w1 = np.random.rand(dim)
  w0 = np.random.rand()
  return w1, w0

def compute_cost(X,Y, y_hat):
    m = len(Y)
    cost = (1/(2*m)) * np.sum(np.square(y_hat - Y))
    return cost


def predict_y(X,w1,w0):
  if len(w1)==1:
    w1 = w1[0]
    return X*w1+w0
  return np.dot(X,w1)+w0

def update_parameters(X,Y,y_hat,cost,w0,w1,learning_rate):
  m = len(Y)
  db=(np.sum(y_hat-Y))/m
  dw=np.dot(y_hat-Y,X)/m

  w0_new=w0-learning_rate*db
  w1_new=w1-learning_rate*dw
  return w0_new,w1_new


def run_gradient_descent(X,Y,alpha,max_iterations,stopping_threshold = 1e-6):
  dims = 1
  if len(X.shape)>1:
    dims = X.shape[1]
  w1,w0=initialize(dims)
  previous_cost = None
  cost_history = np.zeros(max_iterations)
  for itr in range(max_iterations):
    y_hat=predict_y(X,w1,w0)
    cost=compute_cost(X,Y,y_hat)
    # early stopping criteria
    if previous_cost and abs(previous_cost-cost)<=stopping_threshold:
      break
    cost_history[itr]=cost
    previous_cost = cost
    old_w1=w1
    old_w0=w0
    w0,w1=update_parameters(X,Y,y_hat,cost,old_w0,old_w1,alpha)

  return w0,w1,cost_history

X = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787,
  55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
  45.41973014, 54.35163488, 44.1640495 , 58.16847072, 56.72720806,
  48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
Y = np.array([31.70700585, 68.77759598, 62.5623823 , 71.54663223, 87.23092513,
  78.21151827, 79.64197305, 59.17148932, 75.3312423 , 71.30087989,
  55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
  60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])

learning_rate = 0.0001
iterations = 10
print(X.shape,Y.shape)

w0,w1,cost_history = run_gradient_descent(X,Y,learning_rate,iterations)
print(w0,w1)

import numpy as np
import matplotlib.pyplot as plt
# Plot the cost history
plt.plot(range(1, iterations + 1), cost_history, color='blue')
plt.rcParams["figure.figsize"] = (10, 6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')
plt.show()