from matplotlib import pyplot as plt
import numpy as np
from DataGenerator import generateData
from Settings import (DEFAULT_NUMBER_OF_CLASSES,
                        DEFAULT_NUMBER_OF_FEATURES,
                        DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                        DEFAULT_FEATURE_MEAN_RANGE,
                        DEFAULT_RANDOM_NUMBER_SEED,
                        DEFAULT_TRUE_NUMBER_OF_CLASSES)

data, labels = generateData(DEFAULT_NUMBER_OF_CLASSES, DEFAULT_NUMBER_OF_FEATURES,
                            DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                            DEFAULT_FEATURE_MEAN_RANGE, DEFAULT_RANDOM_NUMBER_SEED,
                            DEFAULT_TRUE_NUMBER_OF_CLASSES)

distinctTrainLabels = np.unique(labels)

OPACITY = 0.7
plt.figure()
plt.title("Data Set")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
for i, label in enumerate(distinctTrainLabels):
    plt.scatter(data[labels==label,0], data[labels==label,1],
                c=np.random.rand(3,), alpha=OPACITY,
                label="Class {}".format(i))
plt.legend()            
plt.show()                