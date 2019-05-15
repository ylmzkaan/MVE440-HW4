from matplotlib import pyplot as plt
from sklearn.manifold import MDS
import numpy as np
from DataGenerator import generateData
from Settings import (DEFAULT_NUMBER_OF_CLASSES,
                        DEFAULT_NUMBER_OF_FEATURES,
                        DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                        DEFAULT_FEATURE_MEAN_RANGE,
                        DEFAULT_RANDOM_NUMBER_SEED,
                        DEFAULT_TRUE_NUMBER_OF_CLASSES)

data, labels = generateData(DEFAULT_NUMBER_OF_CLASSES, 
                            DEFAULT_NUMBER_OF_FEATURES,
                            DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                            DEFAULT_FEATURE_MEAN_RANGE, 
                            DEFAULT_RANDOM_NUMBER_SEED,
                            DEFAULT_TRUE_NUMBER_OF_CLASSES)

mds = MDS(n_components=2, n_jobs=-1)
mdsData = mds.fit_transform(data)

distinctTrainLabels = np.unique(labels)
classColors = [np.random.rand(1,3) for _ in distinctTrainLabels]
OPACITY = 0.7

plt.subplot(1, 2, 1)
plt.title("Data Set With MDS")
plt.xlabel("MDS Feature 1")
plt.ylabel("MDS Feature 2")
for i, label in enumerate(distinctTrainLabels):
    plt.scatter(mdsData[labels==label,0], mdsData[labels==label,1],
                c=classColors[i], alpha=OPACITY,
                label="Class {}".format(i))
plt.legend()  

plt.subplot(1, 2, 2)
plt.title("Data Set")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
for i, label in enumerate(distinctTrainLabels):
    plt.scatter(data[labels==label,0], data[labels==label,1],
                c=classColors[i], alpha=OPACITY,
                label="Class {}".format(i))
plt.legend()  
plt.show()                