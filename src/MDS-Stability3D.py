from matplotlib import pyplot as plt
from sklearn.manifold import MDS
import numpy as np
import os
from DataGenerator import generateOneClusterData
from Settings import (DEFAULT_NUMBER_OF_FEATURES,
                        DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                        DEFAULT_FEATURE_MEAN_RANGE)

nPerturbations = 10
nDataSets = 10
nDifferentMdsComponentNumber = 10
mdsNumberOfComponentsRange = range(1, 1+nDifferentMdsComponentNumber)
meanStressDiff = np.zeros((nDifferentMdsComponentNumber,nDataSets))
stdStressDiff = np.zeros((nDifferentMdsComponentNumber,nDataSets))

stressDataDir = os.path.join("data", "MDS-meanStressForDifferentComponentNumber.npy")
normalStress = np.load(stressDataDir)

for j, mdsNumberOfComponents in enumerate(mdsNumberOfComponentsRange):
    print("MDS Number of Components: {}".format(mdsNumberOfComponents))

    for randomNumberSeed in range(nDataSets):
        sampleIdxToDelete = np.random.choice(DEFAULT_NUMBER_OF_RECORDS_PER_CLASS, nPerturbations, replace=False)

        data = generateOneClusterData(DEFAULT_NUMBER_OF_FEATURES,
                                        DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                                        DEFAULT_FEATURE_MEAN_RANGE,
                                        randomNumberSeed,
                                        distribution="normal")

        stressDiff = np.zeros((nPerturbations,))
        for i in range(nPerturbations):

            dataToProcess = np.delete(data, sampleIdxToDelete[i], axis=0)

            mds = MDS(n_components=mdsNumberOfComponents, n_jobs=-1)
            mdsData = mds.fit_transform(dataToProcess)
            stressDiff[i] = mds.stress_ - normalStress[randomNumberSeed,j]

        meanStressDiff[j,randomNumberSeed] = np.mean(stressDiff)
        stdStressDiff[j,randomNumberSeed] = np.std(stressDiff)

plt.figure()
for i in range(nDataSets):
    plt.errorbar(range(1, nDifferentMdsComponentNumber+1), meanStressDiff[:,i],
                yerr=stdStressDiff[:,i], capthick=2, capsize=5,
                label="Data Set: {}".format(i))
plt.xlabel("MDS Component Number")
plt.ylabel("Stress Deviation After Small Perturbation")
plt.title("Mean Stress Deviation After Perturbation")
plt.legend()
plt.show()
