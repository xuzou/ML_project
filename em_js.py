import numpy as np
import os
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.io as scp
from scipy.stats import norm
from sys import maxint
from em_js_alg import expectation, maximization, prob, distance

global shift
global epsilon
global iters
global df_copy
global params
global data_dict

fig = plt.figure()
ax = plt.axes()

# set random seed
rand.seed(42)

mat = scp.loadmat(os.getcwd() + '/data/dataset1.mat')

data_size = len(mat['data'][0])

data = np.matrix(mat['data'])

# Calculate mean and variance
mean = np.mean(data, axis=0)
var = np.var(data, axis=0)

x_values = data[:,:1]
x_data= np.squeeze(np.asarray(x_values))

y_values = data[:,1:2]
y_data= np.squeeze(np.asarray(y_values))


data_dict = {'x': x_data, 'y': y_data, 'label': []}
rand_labels = map(lambda x: x+1, np.random.choice(2, len(data_dict['y'])))
data_dict['label'] = rand_labels


df = pd.DataFrame(data=data_dict)

#fig = plt.figure()
#plt.scatter(data_dict['x'], data_dict['y'], 24, c=data_dict['label'])
#fig.savefig("true-values.png")


### Expectation-maximization

# initial guesses - intentionally bad
guess = { 'mu1': [data_dict['x'][300], data_dict['y'][300]],
           'sig1': [ [1, 0], [0, 1] ],
           'mu2': [data_dict['x'][40], data_dict['y'][40]],
           'sig2': [ [1, 0], [0, 1] ],
           'lambda': [0.5, 0.5]
         }



# loop until parameters converge
shift = maxint
epsilon = 0.01
iters = 0
df_copy = df.copy()
# randomly assign points to their initial clusters
df_copy['label'] = map(lambda x: x+1, np.random.choice(2, len(df)))

ax.scatter(data_dict['x'], data_dict['y'], c = data_dict['label'], cmap=plt.cm.spring)
ax.text(1, 7, "random initialization", fontsize=18)
params = pd.DataFrame(guess)

def main_func():
  global shift
  global epsilon
  global iters
  global df_copy
  global params
  global data_dict
  if shift > epsilon:
    iters += 1
    # E-step
    updated_labels = expectation(df_copy.copy(), params)

    # M-step
    updated_parameters = maximization(updated_labels, params.copy())

    # see if our estimates of mu have changed
    # could incorporate all params, or overall log-likelihood
    shift = distance(params, updated_parameters)

    # logging
    print("iteration {}, shift {}".format(iters, shift))

    # update labels and params for the next iteration
    df_copy = updated_labels
    params = updated_parameters

    #fig = plt.figure()
    #plt.scatter(df_copy['x'], df_copy['y'], 24, c=df_copy['label'])
    #fig.savefig("iteration{}.png".format(iters))

    data_dict['label'] = updated_labels['label']
    return df_copy

def init():
  return

def animate(i):
  global iters
  ax.cla()
  main_func()
  ax.scatter(data_dict['x'], data_dict['y'], c=data_dict['label'], cmap=plt.cm.spring)
  #ax.title('Expectation Maximization')
  tt = "Iteration {}".format(iters)
  print tt
  ax.text(2, 7, tt, fontsize=18)

while shift > epsilon:
  ani = anim.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init, interval=250, blit=False)
  plt.title('Expectation Maximization')
  plt.show()
