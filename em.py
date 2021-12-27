import numpy as np
import random 

from scipy.stats import multivariate_normal as norm
from tabulate import tabulate


MEANS = np.array([[0], [5], [10]])
VARIANCES15 = np.array(([1.5], [1.5], [1.5]))
VARIANCES1 = np.array(([1.0], [1.0], [1.0]))
VARIANCES05 = np.array(([0.5], [0.5], [0.5]))
K = 3


def getData(means, cov):
    data = np.zeros((150, 1))
    i = 0
    while i < 150:
        data[i] = random.gauss(means[i % 3], cov[i % 3])
        i += 1
    return data


class GuassianMixtureModel(object):
    def __init__(self, seed=4, tolerance=1e-6, k=K):
        self.tolerance = tolerance
        self.k = k
        self.weights = np.full(self.k, 1 / self.k)
        np.random.seed(seed)

    def train(self, trainData, iters):
        rows, cols = trainData.shape
        self.resp = np.zeros((rows, self.k))

        choosen = np.random.choice(rows, self.k, replace=False)
        self.means = trainData[choosen]
        self.covs = np.full((self.k, cols, cols), np.cov(trainData, rowvar=False))

        logLikelihood = 0
        for i in range(iters):
            logLikelihoodNew = self._eStep(trainData)
            self._mStep(trainData)
            if abs(logLikelihoodNew - logLikelihood) <= self.tolerance:
                break
            logLikelihood = logLikelihoodNew
        return i+1
    
    def _eStep(self, trainData):
        for i in range(self.k):
            prior = self.weights[i]
            likelihood = norm(self.means[i], self.covs[i]).pdf(trainData)
            self.resp[:, i] = prior * likelihood
        logLikelihood = np.sum(np.log(np.sum(self.resp, axis=1)))
        self.resp = self.resp / self.resp.sum(axis=1, keepdims=1)
        return logLikelihood
    
    def _mStep(self, trainData):
        respWeights = self.resp.sum(axis = 0)
        self.weights = respWeights  / trainData.shape[0]
        weightedSum = np.dot(self.resp.T, trainData)
        self.means = weightedSum / respWeights.reshape(-1, 1)
        for i in range(self.k):
            diff = (trainData - self.means[i]).T
            weightedSum = np.dot(self.resp[:, i] * diff, diff.T)
            self.covs[i] = weightedSum / respWeights[i]

    
if __name__ == "__main__":
    print()
    gmmObj = GuassianMixtureModel()
    trainData = getData(MEANS, VARIANCES05)
    i = gmmObj.train(trainData, 150)
    print("Iteration: {}".format(i))
    print()
    meanInfo = []
    for i in range(3):
        meanInfo.append([i+1, gmmObj.means[i][0], gmmObj.covs[i][0][0]])
    print(tabulate(meanInfo, headers=["Sr. No", "Mean", "Variance"]))
    print()


    trainData = getData(MEANS, VARIANCES1)
    i = gmmObj.train(trainData, 150)
    print("Iteration: {}".format(i))
    print()
    meanInfo = []
    for i in range(3):
        meanInfo.append([i+1, gmmObj.means[i][0], gmmObj.covs[i][0][0]])
    print(tabulate(meanInfo, headers=["Sr. No", "Mean", "Variance"]))
    print()


    trainData = getData(MEANS, VARIANCES15)
    i = gmmObj.train(trainData, 500)
    print("Iteration: {}".format(i))
    print()
    meanInfo = []
    for i in range(3):
        meanInfo.append([i+1, gmmObj.means[i][0], gmmObj.covs[i][0][0]])
    print(tabulate(meanInfo, headers=["Sr. No", "Mean", "Variance"]))
    print()
