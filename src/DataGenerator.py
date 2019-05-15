import numpy as np
from Initialization import initClasses

def generateOneClusterData(numberOfFeatures, numberOfRecordsPerClass,
                 featureMeanRange, randomNumberSeed, distribution="normal"):
    
    featureDistributionData = initClasses(1, numberOfFeatures,
                                        featureMeanRange,
                                        randomNumberSeed=randomNumberSeed)[0]

    if distribution.lower() == "uniform":
        lowerRange = featureDistributionData.featureMeans
        upperRange = 3*featureDistributionData.featureMeans
        data = np.random.uniform(lowerRange, upperRange, size=(numberOfRecordsPerClass, numberOfFeatures))
    elif distribution.lower() == "normal":
        data = np.random.multivariate_normal(featureDistributionData.featureMeans,
                                            featureDistributionData.featureCovariances,
                                            size=(numberOfRecordsPerClass))
    else:      
        raise Exception("Distribution: " + distribution + " is not implemented!")  
                
    return data