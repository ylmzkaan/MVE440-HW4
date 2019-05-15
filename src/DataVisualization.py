from matplotlib import pyplot as plt
import numpy as np
from DataGenerator import generateData
from Settings import (DEFAULT_NUMBER_OF_FEATURES,
                        DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                        DEFAULT_FEATURE_MEAN_RANGE,
                        DEFAULT_RANDOM_NUMBER_SEED)

data = generateOneClusterData(DEFAULT_NUMBER_OF_FEATURES,
                            DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                            DEFAULT_FEATURE_MEAN_RANGE, 
                            DEFAULT_RANDOM_NUMBER_SEED)


OPACITY = 0.7
plt.figure()
plt.title("Data Set")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.scatter(data[:,0], data[:,1],
            c=np.random.rand(1,3), alpha=OPACITY)
plt.legend()            
plt.show()                