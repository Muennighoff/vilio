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
    parser.add_argument('--subdata', action='store_const', default=False, const=True)
    
    # Parse the arguments.
    args = parser.parse_args()

    return args

### FUNCTIONS IMPLEMENTING ENSEMBLE METHODS ###

### HELPERS ###

# Optimizing accuracy based on ROC AUC 
# Source: https://albertusk95.github.io/posts/2019/12/best-threshold-maximize-accuracy-from-roc-pr-curve/
# ACC = (TP + TN)/(TP + TN + FP + FN) = (TP + TN) / P + N   (= Correct ones / all)
# Senstivity / tpr = TP / P 
# Specificity / tnr = TN / N

def get_acc_and_best_threshold_from_roc_curve(tpr, fpr, thresholds, num_pos_class, num_neg_class):

    tp = tpr * num_pos_class
    tn = (1 - fpr) * num_neg_class
    acc = (tp + tn) / (num_pos_class + num_neg_class)

    best_threshold = thresholds[np.argmax(acc)]

    return np.amax(acc), best_threshold

def set_acc(row, threshold):
    if row['proba'] >= threshold:
        val = 1
    else:
        val = 0
    return val


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


def rank_average(subs, weights=None):
    """
    subs: list of submission dataframes with two columns (id, value)
    weights: per submission weights; default is equal weighting 
    """
    if weights is None:
        weights = len(subs) * [1.0 / len(subs)]
    else:
        weights = weights / np.sum(weights)
    preds = subs[0].copy()
    preds.iloc[:,1] = np.zeros(len(subs[0]))
    for i, sub in enumerate(subs):
        preds.iloc[:,1] = np.add(preds.iloc[:,1], weights[i] * rankdata(sub.iloc[:,1]) / len(sub))
        
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
            print("Included in Simple Average: ", csv)
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


def main(path, gt_path="./data/"):
    """
    Loops through Averaging, Power Averaging, Rank Averaging, Optimization to find the best ensemble.

    path: String to directory with csvs of all models
    For each model there should be three csvs: dev, test, test_unseen

    gt_path: Path to folder with ground truth for dev
    """
    # Ground truth
    dev_df = pd.read_json(os.path.join(gt_path, 'dev_seen.jsonl'), lines=True)

    # Make sure the lists will be ordered, i.e. test[0] is the same model as devs[0]
    dev, test, test_unseen = [], [], []
    dev_probas, test_probas, test_unseen_probas = {}, {}, {} # Never dynamically add to a pd Dataframe

    for csv in sorted(os.listdir(path)):
        print(csv)
        if ".csv" in csv:
            if ("dev" in csv) or ("val" in csv):
                dev.append(pd.read_csv(os.path.join(path, csv)))
                dev_probas[csv[:-8]] = pd.read_csv(os.path.join(path, csv)).proba.values
            elif "test_unseen" in csv:
                test_unseen.append(pd.read_csv(os.path.join(path, csv)))
                test_unseen_probas[csv[:-14]] = pd.read_csv(os.path.join(path, csv)).proba.values
            elif "test" in csv:
                test.append(pd.read_csv(os.path.join(path, csv)))
                test_probas[csv[:-7]] = pd.read_csv(os.path.join(path, csv)).proba.values


    dev_probas = pd.DataFrame(dev_probas)
    test_probas = pd.DataFrame(test_probas)
    test_unseen_probas = pd.DataFrame(test_unseen_probas)

    dev_or = dev.copy()
    test_or = test.copy()
    test_unseen_or = test_unseen.copy()

    if len(dev_df) > len(dev_probas):

        print("Your predictions do not include the full dev!")        
        dev_df = dev[0][["id"]].merge(dev_df, how="left", on="id")

    loop, last_score, delta = 0, 0, 0.1

    while (delta > 0.0001):

        # Individual Roc Aucs
        print("Individual RCs:\n")
        print("dev")

        for i, column in enumerate(dev_probas):
            score = roc_auc_score(dev_df.label, dev_probas.iloc[:, i])
            print(column, score)

        print('-'*50)

        if loop > 0:
            while len(dev) > 5:
                lowest_score = 1
                drop = 0
                for i, column in enumerate(dev_probas):
                    score = roc_auc_score(dev_df.label, dev_probas.iloc[:, i])
                    if score < lowest_score:
                        lowest_score = score
                        col = column
                        drop = i

                column_numbers = [x for x in range(dev_probas.shape[1])]  # list of columns' integer indices
                column_numbers.remove(drop)
                dev_probas = dev_probas.iloc[:, column_numbers]

                column_numbers = [x for x in range(test_probas.shape[1])]  # list of columns' integer indices
                column_numbers.remove(drop)
                test_probas = test_probas.iloc[:, column_numbers]

                column_numbers = [x for x in range(test_unseen_probas.shape[1])]  # list of columns' integer indices
                column_numbers.remove(drop)
                test_unseen_probas = test_unseen_probas.iloc[:, column_numbers]
    
                if i < len(dev_or):
                    dev_or.pop(drop)
                    test_or.pop(drop)
                    test_unseen_or.pop(drop)
                if i < len(dev):
                    dev.pop(drop)
                    test.pop(drop)
                    test_unseen.pop(drop)
    
                print("Dropped:", col)
                
        # Spearman Correlations: 
        print("Spearman Corrs:")
        dev_corr = dev_probas.corr(method='spearman')
        test_corr = test_probas.corr(method='spearman')
        test_unseen_corr = test_unseen_probas.corr(method='spearman')
        
        print(dev_corr,'\n')
        print(test_corr)
        print(test_unseen_corr)
        print('-'*50)

        ### SIMPLE AVERAGE ###
        dev_SA = simple_average(dev_probas, dev[0], power=1, normalize=True)
        test_SA = simple_average(test_probas, test[0], power=1, normalize=True)
        test_unseen_SA = simple_average(test_unseen_probas, test_unseen[0], power=1, normalize=True)

        print(roc_auc_score(dev_df.label, dev_SA.proba), accuracy_score(dev_df.label, dev_SA.label))
        print('-'*50)

        ### POWER AVERAGE ###
        dev_PA = simple_average(dev_probas, dev[0], power=2, normalize=True)
        test_PA = simple_average(test_probas, test[0], power=2, normalize=True)
        test_unseen_PA = simple_average(test_unseen_probas, test_unseen[0], power=2, normalize=True)

        print(roc_auc_score(dev_df.label, dev_PA.proba), accuracy_score(dev_df.label, dev_PA.label))
        print('-'*50)

        ### RANK AVERAGE ###
        dev_RA = rank_average(dev)
        test_RA = rank_average(test)
        test_unseen_RA = rank_average(test_unseen)

        print(roc_auc_score(dev_df.label, dev_RA.proba), accuracy_score(dev_df.label, dev_RA.label))
        print('-'*50)

        ### SIMPLEX ###
        weights_dev = Simplex(dev_probas, dev_df.label)

        dev_SX = simple_average(dev_probas, dev[0], weights_dev)
        test_SX = simple_average(test_probas, test[0], weights_dev)
        test_unseen_SX = simple_average(test_unseen_probas, test_unseen[0], weights_dev)

        print(roc_auc_score(dev_df.label, dev_SX.proba), accuracy_score(dev_df.label, dev_SX.label))
        print('-'*50)

        # Prepare Next Round
        dev = dev_or + [dev_SA, dev_PA, dev_RA, dev_SX]
        test = test_or + [test_SA, test_PA, test_RA, test_SX]
        test_unseen = test_unseen_or + [test_unseen_SA, test_unseen_PA, test_unseen_RA, test_unseen_SX]
        
        dev_probas = pd.concat([df.proba for df in dev], axis=1)
        test_probas = pd.concat([df.proba for df in test], axis=1)
        test_unseen_probas = pd.concat([df.proba for df in test_unseen], axis=1)

        # Calculate Delta & increment loop
        delta = abs(roc_auc_score(dev_df.label, dev_SX.proba) - last_score)
        last_score = roc_auc_score(dev_df.label, dev_SX.proba)

        loop += 1

        # I found the loop to not add any value after 2 rounds.
        if loop == 2:
            break
    
    print("Currently at {} after {} loops.".format(last_score, loop))

    # Get accuracy thresholds & optimize (This does not add value to the roc auc, but just to also have an acc score)
    fpr, tpr, thresholds = metrics.roc_curve(dev_df.label, dev_SX.proba)
    acc, threshold = get_acc_and_best_threshold_from_roc_curve(tpr, fpr, thresholds, 250, 250)
    test_SX.label = test_SX.apply(set_acc, axis=1, args=[threshold])
    test_unseen_SX.label = test_unseen_SX.apply(set_acc, axis=1, args=[threshold])

    # Set path instd of /k/w ; Remove all csv data / load the exact same 3 files again as put out
    # As Simplex at some point simply weighs the highest of all - lets take sx as the final prediction after x loops
    dev_SX.to_csv(os.path.join(path, "FIN_dev_seen_" + args.exp + "_" + str(loop) + ".csv"), index=False)
    test_SX.to_csv(os.path.join(path, "FIN_test_seen_" + args.exp + "_" + str(loop) + ".csv"), index=False)
    test_unseen_SX.to_csv(os.path.join(path, "FIN_test_unseen_" + args.exp + "_" + str(loop) + ".csv"), index=False)

    print("Finished.")
  
if __name__ == "__main__":

    args = parse_args()
    
    if args.enstype == "loop":
        main(args.enspath)
    elif args.enstype == "sa":
        sa_wrapper(args.enspath)
    else:
        print(args.enstype, " is not yet enabled. Feel free to add the code :)")
