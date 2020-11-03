import pandas as pd
import numpy as np
import os

from param import args

from sklearn.metrics import roc_auc_score

### SIMPLEX ###

### Similar to scipy optimize
# https://github.com/chrisstroemel/Simple
# https://www.kaggle.com/daisukelab/optimizing-ensemble-weights-using-simple/data

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

def Simplex(dev, devs, label, df_list=False):
    """
    devs: list of dataframes with "proba" column
    label: list/np array of ground truths
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
    
    #def roc_auc(weights):
    #    ''' Will pass the weights as a numpy array '''
    #    final_prediction = 0
    #    for weight, prediction in zip(weights, predictions):
    #            final_prediction += weight*prediction
    #    return roc_auc_score(label, (dev["proba"] + final_prediction))

    def roc_auc(weights):
        ''' Will pass the weights as a numpy array '''
        final_prediction = 0
        for weight, prediction in zip(weights, predictions):
                final_prediction += weight*prediction
        return roc_auc_score(label, final_prediction)

    #def roc_auc(weights):
    #    '''RANK AVERAGE OPTIMIZATION'''
    #    final_prediction = 0
    #    for weight, prediction in zip(weights, predictions):
    #        final_prediction += weight * (rankdata(prediction) / len(prediction))
    #    return roc_auc_score(label, final_prediction)
   


    # This defines the search area, and other optimization parameters.
    # For 11 models, we have 12 corner points -- e.g. all none, only model 1, all others none, only model 2 all others none..
    # We concat an identity matrix & a zero array to create those
    zero_vtx = np.zeros((1, len(predictions)), dtype=int)
    optimization_domain_vertices = np.identity(len(predictions), dtype=int) * 50 # Times 50 here to explore how many times to add; we don't want 0 - 1 vals

    optimization_domain_vertices = np.concatenate((zero_vtx, optimization_domain_vertices), axis=0).tolist()

    
    number_of_iterations = 3000 # Do not put too many or it will overfit
    exploration = 1 # optional, default 0.15

    # Optimize weights
    tuner = SimpleTuner(optimization_domain_vertices, roc_auc, exploration_preference=exploration)
    tuner.optimize(number_of_iterations)
    best_objective_value, best_weights = tuner.get_best()

    print('Optimized =', best_objective_value) # == roc_auc(best_weights)
    print('Weights =', best_weights)

    return best_weights


def main(path):
    """
    Takes in a path with the following expected:
    
    dev (preds)
    dev_IC
    dev_TC
    dev_OC

    test (preds)
    test_IC
    test_TC
    test_OC

    test_unseen (preds)
    test_unseen_IC
    test_unseen_TC
    test_unseen_OC
    
    It then creates the updated dev & tests based on optimizing with IC, TC & OC.
    """

    print("PATH: ", path)

    ### 1/3 LOADING ###

    # ADAPTED TO COMPARE WITH NEW IMPLEMENTATION

    # GT & bases
    dev_GT = pd.read_json("./data/dev_seen.jsonl", lines=True, orient="records")
    test = pd.read_json("./data/test_seen.jsonl", lines=True, orient="records")
    test_unseen = pd.read_json("./data/test_unseen.jsonl", lines=True, orient="records")

    # IC & Full Preds
    for csv in sorted(os.listdir(path)):
        if "dev" in csv:
            if "ic" in csv:
                dev_IC = pd.read_csv(path + csv)
            elif "tc" in csv:
                dev_TC = pd.read_csv(path + csv)
            elif "oc" in csv:
                dev_OC = pd.read_csv(path + csv)
            else:
                dev_ALL = pd.read_csv(path + csv)
        elif "test_unseen" in csv:
            if "ic" in csv:
                test_unseen_IC = pd.read_csv(path + csv)
            elif "tc" in csv:
                test_unseen_TC = pd.read_csv(path + csv)
            elif "oc" in csv:
                test_unseen_OC = pd.read_csv(path + csv)
            else:
                test_unseen_ALL = pd.read_csv(path + csv)
        elif "test" in csv:
            if "ic" in csv:
                test_IC = pd.read_csv(path + csv)
            elif "tc" in csv:
                test_TC = pd.read_csv(path + csv)
            elif "oc" in csv:
                test_OC = pd.read_csv(path + csv)
            else:
                test_ALL = pd.read_csv(path + csv)

    # ALL versions
    dev_IC_ALL = pd.read_json("./data/dev_seen_ic.jsonl", lines=True, orient="records")
    test_IC_ALL = pd.read_json("./data/test_seen_ic.jsonl", lines=True, orient="records")
    test_unseen_IC_ALL = pd.read_json("./data/test_unseen_ic.jsonl", lines=True, orient="records")

    dev_TC_ALL = pd.read_json("./data/dev_seen_tc.jsonl", lines=True, orient="records")
    test_TC_ALL = pd.read_json("./data/test_seen_tc.jsonl", lines=True, orient="records")
    test_unseen_TC_ALL = pd.read_json("./data/test_unseen_tc.jsonl", lines=True, orient="records")

    dev_OC_ALL = pd.read_json("./data/dev_seen_oc.jsonl", lines=True, orient="records")
    test_OC_ALL = pd.read_json("./data/test_seen_oc.jsonl", lines=True, orient="records")
    test_unseen_OC_ALL = pd.read_json("./data/test_unseen_oc.jsonl", lines=True, orient="records")


    ### 2/3 PREPARATION ###

    # 1/3 PREDICTED VERSIONS #
    # IC:
    dev_IC["proba_ic"] = dev_IC["proba"]
    dev_IC["proba_ic"] = (dev_IC.proba_ic - dev_IC.proba_ic.min())/(dev_IC.proba_ic.max()-dev_IC.proba_ic.min())
    dev_IC.drop(["label", "proba"], axis=1, inplace=True)

    test_IC["proba_ic"] = test_IC["proba"]
    test_IC["proba_ic"] = (test_IC.proba_ic - test_IC.proba_ic.min())/(test_IC.proba_ic.max()-test_IC.proba_ic.min())
    test_IC.drop(["label", "proba"], axis=1, inplace=True)

    test_unseen_IC["proba_ic"] = test_unseen_IC["proba"]
    test_unseen_IC["proba_ic"] = (test_unseen_IC.proba_ic - test_unseen_IC.proba_ic.min())/(test_unseen_IC.proba_ic.max()-test_unseen_IC.proba_ic.min())
    test_unseen_IC.drop(["label", "proba"], axis=1, inplace=True)
    
    # TC:
    dev_TC["proba_tc"] = dev_TC["proba"]
    dev_TC["proba_tc"] = (dev_TC.proba_tc - dev_TC.proba_tc.min())/(dev_TC.proba_tc.max()-dev_TC.proba_tc.min())
    dev_TC.drop(["label", "proba"], axis=1, inplace=True)

    test_TC["proba_tc"] = test_TC["proba"]
    test_TC["proba_tc"] = (test_TC.proba_tc - test_TC.proba_tc.min())/(test_TC.proba_tc.max()-test_TC.proba_tc.min())
    test_TC.drop(["label", "proba"], axis=1, inplace=True)

    test_unseen_TC["proba_tc"] = test_unseen_TC["proba"]
    test_unseen_TC["proba_tc"] = (test_unseen_TC.proba_tc - test_unseen_TC.proba_tc.min())/(test_unseen_TC.proba_tc.max()-test_unseen_TC.proba_tc.min())
    test_unseen_TC.drop(["label", "proba"], axis=1, inplace=True)

    # OC:
    dev_OC["proba_oc"] = dev_OC["proba"]
    dev_OC["proba_oc"] = (dev_OC.proba_oc - dev_OC.proba_oc.min())/(dev_OC.proba_oc.max()-dev_OC.proba_oc.min())
    dev_OC.drop(["label", "proba"], axis=1, inplace=True)

    test_OC["proba_oc"] = test_OC["proba"]
    test_OC["proba_oc"] = (test_OC.proba_oc - test_OC.proba_oc.min())/(test_OC.proba_oc.max()-test_OC.proba_oc.min())
    test_OC.drop(["label", "proba"], axis=1, inplace=True)

    test_unseen_OC["proba_oc"] = test_unseen_OC["proba"]
    test_unseen_OC["proba_oc"] = (test_unseen_OC.proba_oc - test_unseen_OC.proba_oc.min())/(test_unseen_OC.proba_oc.max()-test_unseen_OC.proba_oc.min())
    test_unseen_OC.drop(["label", "proba"], axis=1, inplace=True)
    
    # 2/3 ALL VERSIONS TAKEN FROM THE PRED ON ALL DATAPOINTS #

    # IC (Taking probas from normal pred):
    dev_IC_ALL = dev_IC_ALL.merge(dev_ALL, on="id")
    dev_IC_ALL["proba_ic_all"] = dev_IC_ALL["proba"]
    dev_IC_ALL["proba_ic_all"] = (dev_IC_ALL.proba_ic_all - dev_IC_ALL.proba_ic_all.min())/(dev_IC_ALL.proba_ic_all.max()-dev_IC_ALL.proba_ic_all.min())
    dev_IC_ALL = dev_IC_ALL[["id", "proba_ic_all"]]

    test_IC_ALL = test_IC_ALL.merge(test_ALL, on="id")
    test_IC_ALL["proba_ic_all"] = test_IC_ALL["proba"]
    test_IC_ALL["proba_ic_all"] = (test_IC_ALL.proba_ic_all - test_IC_ALL.proba_ic_all.min())/(test_IC_ALL.proba_ic_all.max()-test_IC_ALL.proba_ic_all.min())
    test_IC_ALL = test_IC_ALL[["id", "proba_ic_all"]]

    test_unseen_IC_ALL = test_unseen_IC_ALL.merge(test_unseen_ALL, on="id")
    test_unseen_IC_ALL["proba_ic_all"] = test_unseen_IC_ALL["proba"]
    test_unseen_IC_ALL["proba_ic_all"] = (test_unseen_IC_ALL.proba_ic_all - test_unseen_IC_ALL.proba_ic_all.min())/(test_unseen_IC_ALL.proba_ic_all.max()-test_unseen_IC_ALL.proba_ic_all.min())
    test_unseen_IC_ALL = test_unseen_IC_ALL[["id", "proba_ic_all"]]

    # TC (Taking probas from normal pred):
    dev_TC_ALL = dev_TC_ALL.merge(dev_ALL, on="id")
    dev_TC_ALL["proba_tc_all"] = dev_TC_ALL["proba"]
    dev_TC_ALL["proba_tc_all"] = (dev_TC_ALL.proba_tc_all - dev_TC_ALL.proba_tc_all.min())/(dev_TC_ALL.proba_tc_all.max()-dev_TC_ALL.proba_tc_all.min())
    dev_TC_ALL = dev_TC_ALL[["id", "proba_tc_all"]]

    test_TC_ALL = test_TC_ALL.merge(test_ALL, on="id")
    test_TC_ALL["proba_tc_all"] = test_TC_ALL["proba"]
    test_TC_ALL["proba_tc_all"] = (test_TC_ALL.proba_tc_all - test_TC_ALL.proba_tc_all.min())/(test_TC_ALL.proba_tc_all.max()-test_TC_ALL.proba_tc_all.min())
    test_TC_ALL = test_TC_ALL[["id", "proba_tc_all"]]

    test_unseen_TC_ALL = test_unseen_TC_ALL.merge(test_unseen_ALL, on="id")
    test_unseen_TC_ALL["proba_tc_all"] = test_unseen_TC_ALL["proba"]
    test_unseen_TC_ALL["proba_tc_all"] = (test_unseen_TC_ALL.proba_tc_all - test_unseen_TC_ALL.proba_tc_all.min())/(test_unseen_TC_ALL.proba_tc_all.max()-test_unseen_TC_ALL.proba_tc_all.min())
    test_unseen_TC_ALL = test_unseen_TC_ALL[["id", "proba_tc_all"]]

    # OC (Taking probas from normal pred):
    dev_OC_ALL = dev_OC_ALL.merge(dev_ALL, on="id")
    dev_OC_ALL["proba_oc_all"] = dev_OC_ALL["proba"]
    dev_OC_ALL["proba_oc_all"] = (dev_OC_ALL.proba_oc_all - dev_OC_ALL.proba_oc_all.min())/(dev_OC_ALL.proba_oc_all.max()-dev_OC_ALL.proba_oc_all.min())
    dev_OC_ALL = dev_OC_ALL[["id", "proba_oc_all"]]

    test_OC_ALL = test_OC_ALL.merge(test_ALL, on="id")
    test_OC_ALL["proba_oc_all"] = test_OC_ALL["proba"]
    test_OC_ALL["proba_oc_all"] = (test_OC_ALL.proba_oc_all - test_OC_ALL.proba_oc_all.min())/(test_OC_ALL.proba_oc_all.max()-test_OC_ALL.proba_oc_all.min())
    test_OC_ALL = test_OC_ALL[["id", "proba_oc_all"]]

    test_unseen_OC_ALL = test_unseen_OC_ALL.merge(test_unseen_ALL, on="id")
    test_unseen_OC_ALL["proba_oc_all"] = test_unseen_OC_ALL["proba"]
    test_unseen_OC_ALL["proba_oc_all"] = (test_unseen_OC_ALL.proba_oc_all - test_unseen_OC_ALL.proba_oc_all.min())/(test_unseen_OC_ALL.proba_oc_all.max()-test_unseen_OC_ALL.proba_oc_all.min())
    test_unseen_OC_ALL = test_unseen_OC_ALL[["id", "proba_oc_all"]]

    # 3/3 Merge into one pred df based on ID & Fillna's with 0 #

    dev = dev_ALL.merge(dev_IC, on="id", how="left").merge(dev_TC, on="id", how="left").merge(dev_OC, on="id", how="left")
    dev = dev.merge(dev_IC_ALL, on="id", how="left").merge(dev_TC_ALL, on="id", how="left").merge(dev_OC_ALL, on="id", how="left")
    dev.fillna(0, inplace=True)

    test = test_ALL.merge(test_IC, on="id", how="left").merge(test_TC, on="id", how="left").merge(test_OC, on="id", how="left")
    test = test.merge(test_IC_ALL, on="id", how="left").merge(test_TC_ALL, on="id", how="left").merge(test_OC_ALL, on="id", how="left")
    test.fillna(0, inplace=True)

    test_unseen = test_unseen_ALL.merge(test_unseen_IC, on="id", how="left").merge(test_unseen_TC, on="id", how="left").merge(test_unseen_OC, on="id", how="left")
    test_unseen = test_unseen.merge(test_unseen_IC_ALL, on="id", how="left").merge(test_unseen_TC_ALL, on="id", how="left").merge(test_unseen_OC_ALL, on="id", how="left")    
    test_unseen.fillna(0, inplace=True)


    ### 3/3 CALCULATONS & OUTPUTTING ###

    # Take the best performing proba for each
    final_probas = []
    final_probas.append("proba")

    ic_score = roc_auc_score(dev_IC.merge(dev_GT, on="id").label, dev_IC.merge(dev_GT, on="id").proba_ic)
    ic_all_score = roc_auc_score(dev_IC_ALL.merge(dev_GT, on="id").label, dev_IC_ALL.merge(dev_GT, on="id").proba_ic_all)

    if ic_score > ic_all_score:
        final_probas.append("proba_ic")
    else:
        final_probas.append("proba_ic_all")

    tc_score = roc_auc_score(dev_TC.merge(dev_GT, on="id").label, dev_TC.merge(dev_GT, on="id").proba_tc)
    tc_all_score = roc_auc_score(dev_TC_ALL.merge(dev_GT, on="id").label, dev_TC_ALL.merge(dev_GT, on="id").proba_tc_all)
    
    if tc_score > tc_all_score:
        final_probas.append("proba_tc")
    else:
        final_probas.append("proba_tc_all")

    oc_score = roc_auc_score(dev_OC.merge(dev_GT, on="id").label, dev_OC.merge(dev_GT, on="id").proba_oc)
    oc_all_score = roc_auc_score(dev_OC_ALL.merge(dev_GT, on="id").label, dev_OC_ALL.merge(dev_GT, on="id").proba_oc_all)
    
    if oc_score > oc_all_score:
        final_probas.append("proba_oc")
    else:
        final_probas.append("proba_oc_all")

    probas_only = dev[final_probas]
    gt_only = dev_GT.label

    print("PICKING: ", final_probas)
    print("SCORES:", ic_score, ic_all_score, tc_score, tc_all_score, oc_score, oc_all_score)

    sx_weights = Simplex(dev, probas_only, gt_only, df_list=False)

    print("STARTING WITH: ", roc_auc_score(dev_GT.label, dev.proba))

    dev["proba"] = (dev[final_probas[0]] * sx_weights[0]) + (dev[final_probas[1]] * sx_weights[1]) + (dev[final_probas[2]] * sx_weights[2]) + (dev[final_probas[3]] * sx_weights[3])
    test["proba"] = (test[final_probas[0]] * sx_weights[0]) + (test[final_probas[1]] * sx_weights[1]) + (test[final_probas[2]] * sx_weights[2]) + (test[final_probas[3]] * sx_weights[3])
    test_unseen["proba"] = (test_unseen[final_probas[0]] * sx_weights[0]) + (test_unseen[final_probas[1]] * sx_weights[1]) + (test_unseen[final_probas[2]] * sx_weights[2]) + (test_unseen[final_probas[3]] * sx_weights[3])

    print("FINISHING WITH: ", roc_auc_score(dev_GT.label, dev.proba))

    # Reduce to necessary cols
    dev = dev[["id", "proba", "label"]]
    test = test[["id", "proba", "label"]]
    test_unseen = test_unseen[["id", "proba", "label"]]

    # Replace dev, test, test_unseen in the dir with updated versions:
    for csv in sorted(os.listdir(path)):
        if "dev" in csv:
            if ("IC" in csv) or ("TC" in csv) or ("OC" in csv):
                os.remove(path + csv)
            else:
                dev.to_csv(path + csv, index=False)
        elif "test_unseen" in csv:
            if ("IC" in csv) or ("TC" in csv) or ("OC" in csv):
                os.remove(path + csv)
            else:
                test_unseen.to_csv(path + csv, index=False)
        elif "test" in csv:
            if ("IC" in csv) or ("TC" in csv) or ("OC" in csv):
                os.remove(path + csv)
            else:
                test.to_csv(path + csv, index=False)

if __name__ == "__main__":
    main(args.enspath)