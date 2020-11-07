# Collection of pandas scripts that may be useful
import numpy as np
import pandas as pd
import os

from PIL import Image
import imagehash

from sklearn.metrics import roc_auc_score


# Image hash functions: 
# https://content-blockchain.org/research/testing-different-image-hash-functions/

def phash(img_path):
    phash = imagehash.phash(Image.open(img_path))
    return phash

def crude_hash(img_path):
    """
    The function generates a hash based on simple comparisons such as dimensions of an image
    """
    img = Image.open(img_path)
    img = np.asarray(img, dtype="int32")    
    # Going down the diagonal from ri up corner until no b/w/g value 
    row_val = 0
    col_val = -1
    try:
        while((not(255 > img[row_val, col_val, 0] > 1)) 
              or (not(img[row_val, col_val, 0] != img[row_val, col_val, 1] != img[row_val, col_val, 2]))):
            row_val += 1
            col_val -= 1    
    except:
        row_val = 0
        col_val = -1
        
        try:
            while(not(255 > img[row_val, col_val, 0] > 1)): 
                row_val += 1
                col_val -= 1  
        except:
            try:
                # It has no 3 channels
                while(not(255 > img[row_val, col_val] > 1)): 
                    row_val += 1
                    col_val -= 1 
                    
            except:
                print("3x Except: ", img_path)
                
            hash_v = str(img.shape[0]) + str(img.shape[1]) + str(img[row_val, col_val]) * 3
            return hash_v
            
    hash_v = str(img.shape[0]) + str(img.shape[1]) + str(img[row_val, col_val, 0]) + str(img[row_val, col_val, 1]) + str(img[row_val, col_val, 2])    
    return hash_v

### Data Cleaning

# The HM Dataset is very noisy:
#  In the first version of the dataset there were many duplicates with conflicting labels
#  In the second version, the conflicting labels have all been resolved, yet the duplicates remain
def clean_data(data_path="./data"):
    """
    Cleans the HM train & dev data.
    Outputs traindev & pretrain data.

    data_path: Path to folder with train.jsonl, dev_unseen.jsonl, dev_seen.jsonl
    """
    # Check if the statement was already run and the necessary data exists:
    if os.path.exists(os.path.join(data_path, "pretrain.jsonl")):
        return
    else:
        print("Preparing...")

    ## Load all files
    train = pd.read_json(os.path.join(data_path, "train.jsonl"), lines=True, orient="records")
    dev_seen = pd.read_json(os.path.join(data_path, "dev_seen.jsonl"), lines=True, orient="records")
    # We validate with dev_seen throughout all experiments, so we only take the new data from dev_unseen add it to train and then discard dev_unseen
    dev_unseen = pd.read_json(os.path.join(data_path,"dev_unseen.jsonl"), lines=True, orient="records")
    dev_unseen = dev_unseen[~dev_unseen['id'].isin(dev_seen.id.values)].copy()

    test_seen = pd.read_json(os.path.join(data_path, "test_seen.jsonl"), lines=True, orient="records")
    test_unseen = pd.read_json(os.path.join(data_path, "test_unseen.jsonl"), lines=True, orient="records")

    ## Clean training data
    df_dict = {'train': train, 'dev_seen': dev_seen, 'dev_unseen': dev_unseen}
    train_dist = pd.concat([df.assign(identity=key) for key,df in df_dict.items()])

    # Hash images
    train_dist['full_path'] = train_dist['img'].apply(lambda x: os.path.join(data_path, str(x)))
    train_dist['hash'] = train_dist['full_path'].apply(lambda x: phash(x))
    train_dist['hash_cr'] = train_dist['full_path'].apply(lambda x: crude_hash(x))

    # Find dups among images & text
    train_dist['img_dup0'] = train_dist.duplicated(subset='hash', keep=False)
    train_dist['img_dup1'] = train_dist.duplicated(subset='hash_cr', keep=False)
    train_dist["txtdup"] = train_dist.duplicated(subset='text', keep=False)

    # Identify 100% dups
    hash_df = train_dist.hash.value_counts().reset_index(name="counter")
    hash_df = hash_df.loc[hash_df['counter'] > 1]
    hash_df["hash"] = hash_df["index"].astype(str)

    rmv_ids = []

    for h in hash_df['index'].astype(str).values:
        hash_group = train_dist.loc[train_dist.hash.astype(str) == h]
        
        txtdup = hash_group.duplicated(subset='text', keep=False).values
        imgdup1 = hash_group.duplicated(subset='hash_cr', keep=False).values
        
        if (True in txtdup) and (True in imgdup1):
            if len(txtdup) == 2:
                if hash_group.label.values[0] == hash_group.label.values[1]:
                    rmv_ids.append(hash_group.id.values[0]) # They are 100% identical, we'll just rmv the first
                else:
                    print("Labs not the same:", hash_group.id.values, hash_group.label.values) # None here
            else:
                # About 15 examples which are in the below ISIS & ADD lists.
                pass

    ISIS_dups = [35097, 97542, 91562, 71368, 29013, 85173, 15072, 1348, 70269, 36804, 68954, 91270, 64781, 96078, 97162, 34518, 17834,
                 31408, 56134, 68231, 98517, 27156, 10793, 82169, 25780, 25913, 95401, 94850, 50624, 92845, 58732]
    ADD_dups = [54981, 71903, 69087]

    rmv_ids.extend(ISIS_dups)
    rmv_ids.extend(ADD_dups)
    
    ## Output all files we need
    
    # a) Pretrain file for ITM & LM pre-training

    pretrain = pd.concat([train, dev_seen, test_seen, dev_unseen, test_unseen])

    # The following ids throw some dimension error when pre-training; we can afford to skip them
    dim_error = [63805, 73026, 16845, 27058]
    pretrain = pretrain[~pretrain['id'].isin(dim_error)]
    # Note: The following is slightly different than I did in the original submission, but it performs in fact better
    pretrain["label"].fillna(0, inplace=True)
    pretrain.to_json(path_or_buf=os.path.join(data_path, "pretrain.jsonl"), orient='records', lines=True)

    # b) Cleaned Train + unused data from dev_unseen (All duplicates are in train, hence the following suffices)
    train = train[~train['id'].isin(rmv_ids)].copy()
    trainclean = pd.concat([train, dev_unseen])
    trainclean.to_json(path_or_buf=os.path.join(data_path, "train.jsonl"), orient='records', lines=True)

    # c) Cleaned Train + unused data from dev_unseen + dev_seen
    traincleandev = pd.concat([train, dev_unseen, dev_seen])
    traincleandev.to_json(path_or_buf=os.path.join(data_path, "traindev.jsonl"), orient='records', lines=True)


def create_subdata(data_path="./data"):
    """
    Splits the data into three equal-sized pots to perform subtraining. 

    data_path: Path to data folder containing all jsonl's & images under /img
    """
    # Check if the statement was already run and the necessary data exists:
    if os.path.exists(os.path.join(data_path, "train_s1.jsonl")):
        return
    else:
        print("Preparing...")

    train = pd.read_json(os.path.join(data_path + '/train.jsonl'), lines=True) # Note: This is the updated train, incl. data from dev_unseen
    dev = pd.read_json(os.path.join(data_path + '/dev_seen.jsonl'), lines=True)
    test = pd.read_json(os.path.join(data_path + '/test_seen.jsonl'), lines=True)
    test_unseen = pd.read_json(os.path.join(data_path + '/test_unseen.jsonl'), lines=True)

    df_dict = {'train': train, 'dev': dev, 'test': test, 'test_unseen': test_unseen}
    full_dist = pd.concat([df.assign(identity=key) for key,df in df_dict.items()])

    # Create full path for easy image loading
    full_dist['full_path'] = full_dist['img'].apply(lambda x: os.path.join(data_path, str(x)))

    full_dist['phash'] = full_dist['full_path'].apply(lambda x: phash(x))
    full_dist['crhash'] = full_dist['full_path'].apply(lambda x: crude_hash(x))

    full_dist["text_clean"] = full_dist["text"].str.replace("'", "")
    full_dist["text_clean"] = full_dist["text"].str.replace('"', '')

    full_dist["text_clean"] = full_dist["text_clean"].astype(str)
    full_dist["phash"] = full_dist["phash"].astype(str)
    full_dist["crhash"] = full_dist["crhash"].astype(str)

    full_dist["text_dups"] = full_dist["text_clean"].apply(lambda x: full_dist.loc[full_dist['text_clean'] == x].id.values)
    full_dist["phash_dups"] = full_dist["phash"].apply(lambda x: full_dist.loc[full_dist['phash'] == x].id.values)
    full_dist["crhash_dups"] = full_dist["crhash"].apply(lambda x: full_dist.loc[full_dist['crhash'] == x].id.values)

    dists = {}
    # Create ic dist to focus on data with similar text
    dists["s1"] = full_dist[full_dist["text_dups"].map(len) > 1].copy()

    # Create tc dist to focus on data with similar images
    dists["s2"] = full_dist[(full_dist["phash_dups"].map(len) + full_dist["crhash_dups"].map(len)) > 2].copy()

    # Create oc dist to focus on all the rest; i.e. on the diverse part
    dists["s3"] = full_dist[~((full_dist['id'].isin(dists["s1"].id.values)) | (full_dist['id'].isin(dists["s2"].id.values)))]

    for i in ["s1", "s2", "s3"]:
    
        train = dists[i].loc[dists[i].identity == "train"][["id", "img", "label", "text"]]
        train.to_json(data_path + '/train_' + i + '.jsonl', lines=True, orient="records")

        dev = dists[i].loc[dists[i].identity == "dev"][["id", "img", "label", "text"]]
        dev.to_json(data_path + '/dev_seen_' + i + '.jsonl', lines=True, orient="records")

        traindev = pd.concat([train, dev])
        traindev.to_json(data_path + '/traindev_' + i + '.jsonl', lines=True, orient="records")

        test = dists[i].loc[dists[i].identity == "test"][["id", "img", "text"]]
        test.to_json(data_path + '/test_seen_' + i + '.jsonl', lines=True, orient="records")

        test_unseen = dists[i].loc[dists[i].identity == "test_unseen"][["id", "img", "text"]]
        test_unseen.to_json(data_path + '/test_unseen_' + i + '.jsonl', lines=True, orient="records")

def double_data(data_path="./data", jsonl="test_unseen.jsonl"):
    """
    Takes the data and pastes it on to the end. This ensure the model uses the whole

    jsonl: json lines file with img entry
    """
    data = ["train", "dev_seen", "traindev", "test_seen", "test_unseen"]

    preds = {}
    for csv in sorted(os.listdir(data_path)):
        if any(d in csv for d in data) and ("jsonl" in csv) and ("long" not in csv):
            df = pd.read_json(os.path.join(data_path, csv), lines=True, orient="records")
            if "test" in csv:
                df["label"] = 0
                df.loc[0, "label"] = 1
            pd.concat([df, df]).to_json(os.path.join(data_path, csv[:-6] + "long" + ".jsonl"), lines=True, orient="records")

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
            if "jsonl" in csv:
                print("JLoading: ", csv)
                preds[[d for d in data if d in csv][0] + [s for s in subdata if s in csv][0] + "gt"] = pd.read_json(os.path.join(path, csv), lines=True, orient="records")
            if ("csv" in csv) and (exp in csv):
                print("Loading: ", csv)
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
        print(i)
        print(scores)
        if len(scores) > 0:
            fin_probas.append("proba" + i + max(scores, key=scores.get))
    
    print("PICKING: ", fin_probas)
    
    # Run optimization
    probas_only = preds["dev_seen"][fin_probas]
    gt_only = preds["dev_seen"][["id"]].merge(preds["dev_seengt"], how="left", on="id").label
    
    if len(gt_only) < len(preds["dev_seengt"].label):
        print("Your predictions do not include the full dev!")

    print("STARTED WITH: ", roc_auc_score(gt_only, preds["dev_seen"].proba))
     
    sx_weights = Simplex(probas_only, gt_only, df_list=False, exploration=1, scale=50)
    
    for d in data:
        preds[d]["proba"] = preds[d][fin_probas[0]] * sx_weights[0]
        for i in range(1, len(fin_probas)):
            preds[d]["proba"] += preds[d][fin_probas[i]] * sx_weights[i]
        
    for csv in sorted(os.listdir(path)):
        if any(d in csv for d in data) and ("csv" in csv):
            if any(s in csv for s in subdata[:3]):
                os.remove(os.path.join(path, csv))
            else:
                preds[d].to_csv(os.path.join(path, csv), index=False)


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

