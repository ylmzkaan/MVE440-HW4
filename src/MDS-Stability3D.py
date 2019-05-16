from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from Preprocessing import optimalClusterCount
from gap_statistic import OptimalK
import numpy as np
from DataGenerator import generateOneClusterData
from Settings import (DEFAULT_NUMBER_OF_FEATURES,
                        DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                        DEFAULT_FEATURE_MEAN_RANGE,
                        DEFAULT_RANDOM_NUMBER_SEED)

defaultStressPerMDSComponentNumber = [11439753, 4675575, 2886672, 1992081, 1487373, 1175917, 964234, 811773, 699457, 610293]
nPerturbations = 20
nDataSets = 10
mdsNumberOfComponentsRange = range(1, 11)
meanStressDiff = np.empty((10,nDataSets))
stdStressDiff = np.empty((10,nDataSets))

for j, mdsNumberOfComponents in enumerate(mdsNumberOfComponentsRange):

    for k in range(nDataSets):
        sampleIdxToDelete = np.random.choice(DEFAULT_NUMBER_OF_RECORDS_PER_CLASS, nPerturbations, replace=False)
        stressDiff = np.empty((nPerturbations,))

        print("MDS Number of Components: {}".format(mdsNumberOfComponents))

        data = generateOneClusterData(DEFAULT_NUMBER_OF_FEATURES,
                                        DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                                        DEFAULT_FEATURE_MEAN_RANGE, 
                                        k,
                                        distribution="normal")

        for i in range(nPerturbations):

            #dataToProcess = data.copy()
            #dataToProcess[sampleIdxToDelete[i]] *= 1.2
            dataToProcess = np.delete(data, sampleIdxToDelete[i], axis=0)

            mds = MDS(n_components=mdsNumberOfComponents, n_jobs=-1)
            mdsData = mds.fit_transform(dataToProcess)
            stressDiff[i] = defaultStressPerMDSComponentNumber[j] - mds.stress_

        meanStressDiff[j,k] = np.mean(stressDiff)
        stdStressDiff[j,k] = np.std(stressDiff)

plt.figure()
for i in range(10):
    plt.errorbar(range(1, 11), meanStressDiff[:,i], yerr=stdStressDiff[:,i],
                capthick=2, capsize=10, linewidth=3, label="Data Set: {}".format(i))  
plt.xlabel("MDS Component Number")
plt.ylabel("Stress Deviation After Small Perturbation")
plt.title("Mean Stress Deviation After Perturbation")
plt.legend()
plt.show()