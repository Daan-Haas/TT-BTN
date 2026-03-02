import matplotlib.pyplot as plt

from utils import *
from model import *
from toy_data import *

seed = 101
np.random.seed(seed)
### Data settings ###
I = 5 #
N = 100 # Number of data points
noise_var = 0.01 # Noise variance

### Model settings ###
D = 3 # Number of cores
ranks = [5 for _ in range(D-1)] # Tensor-train ranks
ranks = [1] + ranks + [1] # first and last rank must be 1 to maintain output dimension
dims = [I for _ in range(D)] # dimensionality of kernels

X_train, Y_train, X_test, Y_test, ground_truth = generate_quadratic_dataset(I, N, 0)
noise = [0.001*np.random.rand(ranks[d], dims[d], ranks[d+1]) for d in range(D)]
model = ground_truth
for d, core in enumerate(model.W.cores):
    core.core = core.core + noise[d]
model.train(X_train, Y_train, iteration_limit=20)

results = model.predict(X_test)

plt.scatter(X_test[:,0], Y_test, label="Ground truth")
plt.scatter(X_test[:,0], results, label="Predicted")
plt.legend()
plt.show()
