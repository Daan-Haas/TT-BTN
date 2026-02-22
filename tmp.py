import matplotlib.pyplot as plt
import numpy as np

covariance = np.eye(8)
covariance = covariance.reshape(2,2,2,2,2,2)
covariance = covariance.transpose([0,3,1,4,2,5])
covariance =covariance.reshape(-1,4)
print(covariance)