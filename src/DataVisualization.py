from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import MDS
from DataGenerator import generateOneClusterData
from Settings import (DEFAULT_NUMBER_OF_FEATURES,
                        DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                        DEFAULT_FEATURE_MEAN_RANGE,
                        DEFAULT_RANDOM_NUMBER_SEED)

data = generateOneClusterData(DEFAULT_NUMBER_OF_FEATURES,
                            DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                            DEFAULT_FEATURE_MEAN_RANGE, 
                            DEFAULT_RANDOM_NUMBER_SEED,
                            distribution="normal")

mds = MDS(n_components=2, n_jobs=-1)
mdsData = mds.fit_transform(data)

OPACITY = 0.7
plt.subplot(1,2,1)
plt.title("Data Set")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.scatter(data[:,0], data[:,1],
            c="blue", alpha=OPACITY)
plt.subplot(1,2,2)
plt.title("Data Set With MDS")
plt.xlabel("MDS Feature 1")
plt.ylabel("MDS Feature 2")
plt.scatter(mdsData[:,0], mdsData[:,1],
            c="orange", alpha=OPACITY)            
plt.show()                