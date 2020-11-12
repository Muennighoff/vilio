import pandas as pd
import numpy as np
import os

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import metrics

from scipy.stats import rankdata

import math

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--enspath", type=str, default="./data", help="Path to folder with all csvs")
    parser.add_argument("--enstype", type=str, default="loop", help="Type of ensembling to be performed - Current options: loop / sa")
    parser.add_argument("--exp", type=str, default="experiment", help="Name of experiment for csv's")
    
    # Parse the arguments.
    args = parser.parse_args()

    return args

### FUNCTIONS IMPLEMENTING ENSEMBLE METHODS ###

### HELPERS ###


### AVERAGES ###

def simple_average(targets, example, weights=None, power=1, normalize=False):
    """
    targets: df with target values as columns
    example: output df example (e.g. including ID - make sure to adjust iloc below if target is not at 1)
    weights: per submission weights; default is equal weighting 
    power: optional for power averaging
    normalize: Whether to normalize targets btw 0 & 1
    """
    if weights is None:
        weights = len(targets.columns) * [1.0 / len(targets.columns)]
    else:
        weights = weights / np.sum(weights)

    preds = example.copy()
    preds.iloc[:,1] = np.zeros(len(preds))

    if normalize:
        targets = (targets - targets.min())/(targets.max()-targets.min())
    for i in range(len(targets.columns)):
        preds.iloc[:,1] = np.add(preds.iloc[:, 1], weights[i] * (targets.iloc[:, i].astype(float)**power))
    
    return preds


### SIMPLEX ###

### Similar to scipy optimize
# Taken & adapted from:
# https://github.com/chrisstroemel/Simple

from heapq import heappush, heappop, heappushpop
import numpy
import math
import time
import matplotlib.pyplot as plotter

CAPACITY_INCREMENT = 1000

class _Simplex:
	def __init__(self, pointIndices, testCoords, contentFractions, objectiveScore, opportunityCost, contentFraction, difference):
		self.pointIndices = pointIndices
		self.testCoords = testCoords
		self.contentFractions = contentFractions
		self.contentFraction = contentFraction
		self.__objectiveScore = objectiveScore
		self.__opportunityCost = opportunityCost
		self.update(difference)

	def update(self, difference):
		self.acquisitionValue = -(self.__objectiveScore + (self.__opportunityCost * difference))
		self.difference = difference

	def __eq__(self, other):
		return self.acquisitionValue == other.acquisitionValue

	def __lt__(self, other):
		return self.acquisitionValue < other.acquisitionValue

class SimpleTuner:
	def __init__(self, cornerPoints, objectiveFunction, exploration_preference=0.15):
		self.__cornerPoints = cornerPoints
		self.__numberOfVertices = len(cornerPoints)
		self.queue = []
		self.capacity = self.__numberOfVertices + CAPACITY_INCREMENT
		self.testPoints = numpy.empty((self.capacity, self.__numberOfVertices))
		self.objective = objectiveFunction
		self.iterations = 0
		self.maxValue = None
		self.minValue = None
		self.bestCoords = []
		self.opportunityCostFactor = exploration_preference #/ self.__numberOfVertices
			

	def optimize(self, maxSteps=10):
		for step in range(maxSteps):
			#print(self.maxValue, self.iterations, self.bestCoords)
			if len(self.queue) > 0:
				targetSimplex = self.__getNextSimplex()
				newPointIndex = self.__testCoords(targetSimplex.testCoords)
				for i in range(0, self.__numberOfVertices):
					tempIndex = targetSimplex.pointIndices[i]
					targetSimplex.pointIndices[i] = newPointIndex
					newContentFraction = targetSimplex.contentFraction * targetSimplex.contentFractions[i]
					newSimplex = self.__makeSimplex(targetSimplex.pointIndices, newContentFraction)
					heappush(self.queue, newSimplex)
					targetSimplex.pointIndices[i] = tempIndex
			else:
				testPoint = self.__cornerPoints[self.iterations]
				testPoint.append(0)
				testPoint = numpy.array(testPoint, dtype=numpy.float64)
				self.__testCoords(testPoint)
				if self.iterations == (self.__numberOfVertices - 1):
					initialSimplex = self.__makeSimplex(numpy.arange(self.__numberOfVertices, dtype=numpy.intp), 1)
					heappush(self.queue, initialSimplex)
			self.iterations += 1

	def get_best(self):
		return (self.maxValue, self.bestCoords[0:-1])

	def __getNextSimplex(self):
		targetSimplex = heappop(self.queue)
		currentDifference = self.maxValue - self.minValue
		while currentDifference > targetSimplex.difference:
			targetSimplex.update(currentDifference)
			# if greater than because heapq is in ascending order
			if targetSimplex.acquisitionValue > self.queue[0].acquisitionValue:
				targetSimplex = heappushpop(self.queue, targetSimplex)
		return targetSimplex
		
	def __testCoords(self, testCoords):
		objectiveValue = self.objective(testCoords[0:-1])
		if self.maxValue == None or objectiveValue > self.maxValue: 
			self.maxValue = objectiveValue
			self.bestCoords = testCoords
			if self.minValue == None: self.minValue = objectiveValue
		elif objectiveValue < self.minValue:
			self.minValue = objectiveValue
		testCoords[-1] = objectiveValue
		if self.capacity == self.iterations:
			self.capacity += CAPACITY_INCREMENT
			self.testPoints.resize((self.capacity, self.__numberOfVertices))
		newPointIndex = self.iterations
		self.testPoints[newPointIndex] = testCoords
		return newPointIndex


	def __makeSimplex(self, pointIndices, contentFraction):
		vertexMatrix = self.testPoints[pointIndices]
		coordMatrix = vertexMatrix[:, 0:-1]
		barycenterLocation = numpy.sum(vertexMatrix, axis=0) / self.__numberOfVertices

		differences = coordMatrix - barycenterLocation[0:-1]
		distances = numpy.sqrt(numpy.sum(differences * differences, axis=1))
		totalDistance = numpy.sum(distances)
		barycentricTestCoords = distances / totalDistance

		euclideanTestCoords = vertexMatrix.T.dot(barycentricTestCoords)
		
		vertexValues = vertexMatrix[:,-1]

		testpointDifferences = coordMatrix - euclideanTestCoords[0:-1]
		testPointDistances = numpy.sqrt(numpy.sum(testpointDifferences * testpointDifferences, axis=1))



		inverseDistances = 1 / testPointDistances
		inverseSum = numpy.sum(inverseDistances)
		interpolatedValue = inverseDistances.dot(vertexValues) / inverseSum


		currentDifference = self.maxValue - self.minValue
		opportunityCost = self.opportunityCostFactor * math.log(contentFraction, self.__numberOfVertices)

		return _Simplex(pointIndices.copy(), euclideanTestCoords, barycentricTestCoords, interpolatedValue, opportunityCost, contentFraction, currentDifference)

	def plot(self):
		if self.__numberOfVertices != 3: raise RuntimeError('Plotting only supported in 2D')
		matrix = self.testPoints[0:self.iterations, :]

		x = matrix[:,0].flat
		y = matrix[:,1].flat
		z = matrix[:,2].flat

		coords = []
		acquisitions = []

		for triangle in self.queue:
			coords.append(triangle.pointIndices)
			acquisitions.append(-1 * triangle.acquisitionValue)


		plotter.figure()
		plotter.tricontourf(x, y, coords, z)
		plotter.triplot(x, y, coords, color='white', lw=0.5)
		plotter.colorbar()


		plotter.figure()
		plotter.tripcolor(x, y, coords, acquisitions)
		plotter.triplot(x, y, coords, color='white', lw=0.5)
		plotter.colorbar()

		plotter.show()

def Simplex(devs, label, df_list=False, exploration=0.01, scale=1):
    """
    devs: list of dataframes with "proba" column
    label: list/np array of ground truths
    scale: By default we will get weights in the 0-1 range. Setting e.g. scale=50, gives weights in the 0-50 range.
    """
    predictions = []
    if df_list:
        for df in devs:
            predictions.append(df.proba)

        print(len(predictions[0]))
    else:
        for i, column in enumerate(devs):
            predictions.append(devs.iloc[:, i])

        print(len(predictions[0]))

    print("Optimizing {} inputs.".format(len(predictions)))

    def roc_auc(weights):
        ''' Will pass the weights as a numpy array '''
        final_prediction = 0
        for weight, prediction in zip(weights, predictions):
                final_prediction += weight*prediction
        return roc_auc_score(label, final_prediction)

    # This defines the search area, and other optimization parameters.
    # For e.g. 11 models, we have 12 corner points -- e.g. all none, only model 1, all others none, only model 2 all others none..
    # We concat an identity matrix & a zero array to create those
    zero_vtx = np.zeros((1, len(predictions)), dtype=int)
    optimization_domain_vertices = np.identity(len(predictions), dtype=int) * scale

    optimization_domain_vertices = np.concatenate((zero_vtx, optimization_domain_vertices), axis=0).tolist()

    
    number_of_iterations = 3000
    exploration = exploration # optional, default 0.01

    # Optimize weights
    tuner = SimpleTuner(optimization_domain_vertices, roc_auc, exploration_preference=exploration)
    tuner.optimize(number_of_iterations)
    best_objective_value, best_weights = tuner.get_best()

    print('Optimized =', best_objective_value) # same as roc_auc(best_weights)
    print('Weights =', best_weights)

    return best_weights

### APPLYING THE HELPER FUNCTIONS ###

import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score

def combine_subdata(path, gt_path="./data/", exp="", subtrain=True):
    """
    Combines predictions from submodels & main model.

    path: String to directory with csvs of all models
    gt_path: Path to folder with ground truth for dev

    exp: Define an exp to make sure we do not load unrelated csvs
    subpreds: Whether to use preds from subtrained or from the pred on full
    """ 

    data = ["dev_seen", "test_seen", "test_unseen"]
    subdata = ["s1", "s2", "s3", ""]
    if subtrain:
        types = ["", "gt"]
    else:
        types = ["gt"]
    
    # Load data
    preds = {}
    for csv in sorted(os.listdir(path)):
        if any(d in csv for d in data):
            if ("jsonl" in csv) and ("long" not in csv):
                preds[[d for d in data if d in csv][0] + [s for s in subdata if s in csv][0] + "gt"] = pd.read_json(os.path.join(path, csv), lines=True, orient="records")
            if ("csv" in csv) and (exp in csv):
                preds[[d for d in data if d in csv][0] + [s for s in subdata if s in csv][0]] = pd.read_csv(os.path.join(path, csv))

    # Normalize probabilities
    for d in data:
        for x in types:
            for i in ["s1", "s2", "s3"]:
                if x == "gt":
                    preds[d+i+x] = preds[d+i+x].merge(preds[d], on="id")
                preds[d+i+x]["proba"+i+x] = preds[d+i+x]["proba"]
                preds[d+i+x]["proba"+i+x] = (preds[d+i+x]["proba"+i+x] - preds[d+i+x]["proba"+i+x].min())/(preds[d+i+x]["proba"+i+x].max()-preds[d+i+x]["proba"+i+x].min())
                preds[d+i+x] = preds[d+i+x][["id", "proba"+i+x]]    
            preds[d+"s4"+x] = preds[d+"s1"+x].merge(preds[d+"s2"+x], on="id", how="inner")
            preds[d+"s4"+x]["proba"+"s4"+x] = (preds[d+"s4"+x]["proba"+"s1"+x] + preds[d+"s4"+x]["proba"+"s2"+x])/2
            preds[d+"s4"+x] = preds[d+"s4"+x][["id", "proba"+"s4"+x]]
            for i in ["s1", "s2"]:
                preds[d+i+x] = preds[d+i+x].loc[~preds[d+i+x].id.isin(preds[d+"s4"+x].id.values)]
        
    # Combine
    for d in data:
        for i in ["s1", "s2", "s3", "s4"]:
            for x in types:
                preds[d] = preds[d].merge(preds[d+i+x], on="id", how="left")
        preds[d].fillna(0, inplace=True)
        
    # Decide on probas
    fin_probas = ["proba"]
    for i in ["s1", "s2", "s3", "s4"]:
        scores = {}
        for x in types:
            df = preds["dev_seen"+i+x].merge(preds["dev_seengt"], on="id")
            if len(df) > 1:
                try: # Fails when only one label present
                    scores[x] = roc_auc_score(df["label"], df["proba"+i+x])
                except:
                    scores[x] = 0
        if len(scores) > 0:
            fin_probas.append("proba" + i + max(scores, key=scores.get))
    
    # Run optimization
    probas_only = preds["dev_seen"][fin_probas]
    gt_only = preds["dev_seen"][["id"]].merge(preds["dev_seengt"], how="left", on="id").label
    
    if len(gt_only) < len(preds["dev_seengt"].label):
        print("Your predictions do not include the full dev!")
     
    sx_weights = Simplex(probas_only, gt_only, df_list=False, exploration=1, scale=50)
    
    for d in data:
        preds[d]["proba"] = preds[d][fin_probas[0]] * sx_weights[0]
        for i in range(1, len(fin_probas)):
            preds[d]["proba"] += preds[d][fin_probas[i]] * sx_weights[i]
        
    for csv in sorted(os.listdir(path)):
        if any(d in csv for d in data) and ("csv" in csv) and (exp in csv):
            if any(s in csv for s in subdata[:3]):
                os.remove(os.path.join(path, csv))
            elif data[0] in csv:
                preds[data[0]][["id", "proba", "label"]].to_csv(os.path.join(path, csv), index=False)
            elif data[1] in csv:
                preds[data[1]][["id", "proba", "label"]].to_csv(os.path.join(path, csv), index=False)
            elif data[2] in csv:
                preds[data[2]][["id", "proba", "label"]].to_csv(os.path.join(path, csv), index=False)


def sa_wrapper(data_path="./data"):
    """
    Applies simple average.

    data_path: path to folder with  X * (dev_seen, test_seen & test_unseen) .csv files
    """
    # Make sure the lists will be ordered, i.e. test[0] is the same model as devs[0]
    dev, test, test_unseen = [], [], []
    dev_probas, test_probas, test_unseen_probas = {}, {}, {} # Never dynamically add to a pd Dataframe

    for csv in sorted(os.listdir(data_path)):
        if ".csv" in csv:
            print("Included in Simple Average: ")
            if ("dev" in csv) or ("val" in csv):
                dev.append(pd.read_csv(data_path + csv))
                dev_probas[csv[:-8]] = pd.read_csv(data_path + csv).proba.values
            elif "test_unseen" in csv:
                test_unseen.append(pd.read_csv(data_path + csv))
                test_unseen_probas[csv[:-14]] = pd.read_csv(data_path + csv).proba.values
            elif "test" in csv:
                test.append(pd.read_csv(data_path + csv))
                test_probas[csv[:-7]] = pd.read_csv(data_path + csv).proba.values

    dev_probas = pd.DataFrame(dev_probas)
    test_probas = pd.DataFrame(test_probas)
    test_unseen_probas = pd.DataFrame(test_unseen_probas)

    dev_SA = simple_average(dev_probas, dev[0])
    test_SA = simple_average(test_probas, test[0])
    test_unseen_SA = simple_average(test_unseen_probas, test_unseen[0])

    # Create output dir
    os.makedirs(os.path.join(data_path, args.exp), exist_ok=True)

    for csv in sorted(os.listdir(data_path)):
        if ".csv" in csv:
            if ("dev" in csv) or ("val" in csv):
                os.remove(os.path.join(data_path, csv))
                dev_SA.to_csv(os.path.join(data_path, args.exp, args.exp + "_dev_seen_SA.csv"), index=False)   
            elif "test_unseen" in csv:
                os.remove(os.path.join(data_path, csv))
                test_unseen_SA.to_csv(os.path.join(data_path, args.exp, args.exp + "_test_unseen_SA.csv"), index=False)   
            elif "test" in csv:
                os.remove(os.path.join(data_path, csv))
                test_SA.to_csv(os.path.join(data_path, args.exp, args.exp + "_test_seen_SA.csv"), index=False)

  
if __name__ == "__main__":

    args = parse_args()
    
    if args.enstype == "sa":
        sa_wrapper(args.enspath)
    else:
        print(args.enstype, " is not yet enabled. Feel free to add the code :)")
