import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def optimalClusterCount(data, clusterCountRange=np.arange(2,10),
                        randomNumberSeed=0):

    nClusterCountToTry = len(clusterCountRange)
    silhouetteScores = np.empty((nClusterCountToTry,), dtype=tuple)

    for i, nClusters in enumerate(clusterCountRange):
        clusterer = KMeans(n_clusters=nClusters, random_state=randomNumberSeed)
        clusterLabels = clusterer.fit_predict(data)
        silhouetteAvg = silhouette_score(data, clusterLabels)
        silhouetteScores[i] = (nClusters, silhouetteAvg)

    optimalClusterCount = max(silhouetteScores, key = lambda x: x[1])[0]
    return optimalClusterCount
