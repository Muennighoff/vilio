import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
from torch import optim
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import metrics

from scipy.stats import rankdata

import math

from param import args

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

import math
import torch
from torch.optim.optimizer import Optimizer, required

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

def Simplex(devs, label, df_list=False):
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
    optimization_domain_vertices = np.identity(len(predictions), dtype=int)

    optimization_domain_vertices = np.concatenate((zero_vtx, optimization_domain_vertices), axis=0).tolist()

    
    number_of_iterations = 3000
    exploration = 0.01 # optional, default 0.15

    # Optimize weights
    tuner = SimpleTuner(optimization_domain_vertices, roc_auc, exploration_preference=exploration)
    tuner.optimize(number_of_iterations)
    best_objective_value, best_weights = tuner.get_best()

    print('Optimized =', best_objective_value) # == roc_auc(best_weights)
    print('Weights =', best_weights)

    return best_weights


### NEURAL NETWORK ###

def roc_star_loss( _y_true, y_pred, gamma, _epoch_true, epoch_pred):
    """
    Nearly direct loss function for AUC.
    See article,
    C. Reiss, "Roc-star : An objective function for ROC-AUC that actually works."
    https://github.com/iridiumblue/articles/blob/master/roc_star.md
        _y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
        y_pred: `Tensor` . Predictions.
        gamma  : `Float` Gamma, as derived from last epoch.
        _epoch_true: `Tensor`.  Targets (labels) from last epoch.
        epoch_pred : `Tensor`.  Predicions from last epoch.
    """
    #convert labels to boolean
    y_true = (_y_true>=0.50)
    epoch_true = (_epoch_true>=0.50)

    # if batch is either all true or false return small random stub value.
    if torch.sum(y_true)==0 or torch.sum(y_true) == y_true.shape[0]: return torch.sum(y_pred)*1e-8

    pos = y_pred[y_true]
    neg = y_pred[~y_true]

    epoch_pos = epoch_pred[epoch_true]
    epoch_neg = epoch_pred[~epoch_true]

    # Take random subsamples of the training set, both positive and negative.
    max_pos = 1000 # Max number of positive training samples
    max_neg = 1000 # Max number of positive training samples
    cap_pos = epoch_pos.shape[0]
    cap_neg = epoch_neg.shape[0]
    epoch_pos = epoch_pos[torch.rand_like(epoch_pos) < max_pos/cap_pos]
    epoch_neg = epoch_neg[torch.rand_like(epoch_neg) < max_neg/cap_pos]

    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]

    # sum positive batch elements agaionst (subsampled) negative elements
    if ln_pos>0 :
        pos_expand = pos.view(-1,1).expand(-1,epoch_neg.shape[0]).reshape(-1)
        neg_expand = epoch_neg.repeat(ln_pos)

        diff2 = neg_expand - pos_expand + gamma
        l2 = diff2[diff2>0]
        m2 = l2 * l2
        len2 = l2.shape[0]
    else:
        m2 = torch.tensor([0], dtype=torch.float).cuda()
        len2 = 0

    # Similarly, compare negative batch elements against (subsampled) positive elements
    if ln_neg>0 :
        pos_expand = epoch_pos.view(-1,1).expand(-1, ln_neg).reshape(-1)
        neg_expand = neg.repeat(epoch_pos.shape[0])

        diff3 = neg_expand - pos_expand + gamma
        l3 = diff3[diff3>0]
        m3 = l3*l3
        len3 = l3.shape[0]
    else:
        m3 = torch.tensor([0], dtype=torch.float).cuda()
        len3=0

    if (torch.sum(m2)+torch.sum(m3))!=0 :
        res2 = torch.sum(m2)/max_pos+torch.sum(m3)/max_neg
       #code.interact(local=dict(globals(), **locals()))
    else:
        res2 = torch.sum(m2)+torch.sum(m3)

    res2 = torch.where(torch.isnan(res2), torch.zeros_like(res2), res2)

    return res2

def epoch_update_gamma(y_true,y_pred, epoch=-1,delta=2.0):
    """
    Calculate gamma from last epoch's targets and predictions.
    Gamma is updated at the end of each epoch.
    y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
    y_pred: `Tensor` . Predictions.
    """
    DELTA = delta
    SUB_SAMPLE_SIZE = 2000.0
    pos = y_pred[y_true==1]
    neg = y_pred[y_true==0] # yo pytorch, no boolean tensors or operators?  Wassap?
    # subsample the training set for performance
    cap_pos = pos.shape[0]
    cap_neg = neg.shape[0]
    pos = pos[torch.rand_like(pos) < SUB_SAMPLE_SIZE/cap_pos]
    neg = neg[torch.rand_like(neg) < SUB_SAMPLE_SIZE/cap_neg]
    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]
    pos_expand = pos.view(-1,1).expand(-1,ln_neg).reshape(-1)
    neg_expand = neg.repeat(ln_pos)
    diff = neg_expand - pos_expand
    ln_All = diff.shape[0]
    Lp = diff[diff>0] # because we're taking positive diffs, we got pos and neg flipped.
    ln_Lp = Lp.shape[0]-1
    diff_neg = -1.0 * diff[diff<0]
    diff_neg = diff_neg.sort()[0]
    ln_neg = diff_neg.shape[0]-1
    ln_neg = max([ln_neg, 0])
    left_wing = int(ln_Lp*DELTA)
    left_wing = max([0,left_wing])
    left_wing = min([ln_neg,left_wing])
    default_gamma=torch.tensor(0.2, dtype=torch.float).cuda()
    if diff_neg.shape[0] > 0 :
        gamma = diff_neg[left_wing]
    else:
        gamma = default_gamma # default=torch.tensor(0.2, dtype=torch.float).cuda() #zoink
    L1 = diff[diff>-1.0*gamma]
    ln_L1 = L1.shape[0]
    if epoch > -1 :
        return gamma
    else :
        return default_gamma


def gelu_new(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return gelu_new(x)

class GeluNN(nn.Module):
    def __init__(self, feats):
        super().__init__()

        self.fc1 = nn.Linear(feats, feats*2)
        self.gelu = GeLU()
        self.norm = nn.LayerNorm(feats*2, eps=1e-12)
        self.fc2 = nn.Linear(feats*2, 1)

        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        :param feat: b, 2, o, f
        :param pos:  b, 2, o, 4
        :param sent: b, (string)
        :param leng: b, (numpy, int)
        :return:
        """

        x = self.fc1(x)
        x = self.gelu(x)
        x = self.norm(x)

        #x = self.dropout(x)

        x = self.fc2(x)

        return x

def nn_train(model, df, label_full, seed=42, quiet=True, epochs=40, batch_size=32, split=0.9, lr=0.005):
    """
    model: torch.nn.Module to train
    df: Pandas dataframe with predictions as columns (these will be used as features)
    label: Labels for each row in df
    """
    # Prepare data:
    split = int(len(df) * split)

    feats = df.values[:split]
    label = label_full[:split]

    feats_val = df.values[split:]
    label_val = label_full[split:]


    np.random.seed(seed)
    torch.manual_seed(seed) 
    
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    feats = torch.from_numpy(feats).float()
    label = torch.from_numpy(label).float()
    
    feats_val = torch.from_numpy(feats_val).float()
    label_val = torch.from_numpy(label_val).float()
    
    model = model.cuda()
    
    last_epoch_y_pred = torch.tensor(1.0-np.random.rand(len(feats))/2.0, dtype=torch.float).cuda()
    last_epoch_y_t = torch.tensor([o for o in label], dtype=torch.float).cuda()
    epoch_gamma = 0.20
    
    dataset_train = TensorDataset(feats, label)
    trainloader = DataLoader(dataset_train, batch_size=batch_size, sampler=RandomSampler(dataset_train))

    # SWA Model
    #t_total = int(len(trainloader) * epochs)
    #swa_model = AveragedModel(model)
    #swa_model = swa_model.cuda()
    #swa_start = t_total * 0.25
    #swa_scheduler = SWALR(optimizer, swa_lr=lr)
    #ups = 0


    for epoch in range(epochs):
        if quiet==False:
            print(" --- EPOCH {} ---\n".format(epoch))

        total_loss = 0.

        epoch_y_pred = []
        epoch_y_t = []

        model.train()
        #swa_model.train()

        for batch in trainloader:
            
            optimizer.zero_grad()

            feats, label = batch
            feats, label = feats.cuda(), label.cuda()

            logit = model(feats)
            logit = logit.squeeze()

            loss = roc_star_loss(label,logit,epoch_gamma, last_epoch_y_t, last_epoch_y_pred)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            #if (ups > swa_start):
            #    swa_model.update_parameters(model)
            #    swa_scheduler.step()

            epoch_y_pred.extend(logit)
            epoch_y_t.extend(label)

            #ups += 1

        if quiet==False:
            print("TRAIN: ", loss.item(), "Exmpls: ", logit.detach()[:3])

        last_epoch_y_pred = torch.tensor(epoch_y_pred).cuda()
        last_epoch_y_t = torch.tensor(epoch_y_t).cuda()
        epoch_gamma = epoch_update_gamma(last_epoch_y_t, last_epoch_y_pred, epoch)

        # VAL
        model.eval()
        #swa_model.eval()

        with torch.no_grad():
            
            feats_val, label_val = feats_val.cuda(), label_val.cuda()
            logit = model(feats_val)
            logit = logit.squeeze()
            loss = criterion(logit, label_val)
            score = roc_auc_score(label_val.detach().cpu().numpy(), logit.detach().cpu().numpy())

            if quiet==False:
                print("DEV: ", loss.item(), "Exmpls: ", logit.detach()[:3], "RC: ", score, "\n")
    
    print("DEV: ", loss.item(), "Exmpls: ", logit.detach()[:3], "RC: ", score, "\n")
    
    return model

def nn_predict(df, csv_files, model):
    feats = torch.from_numpy(df.values).float()
    model.eval()
    with torch.no_grad():
        logit = model(feats.cuda())
        logit = logit.squeeze()

    preds = csv_files[0].copy()
    preds.iloc[:,1] = logit.detach().cpu().numpy()

    return preds


### MAIN FUNCTION - WORKFLOW

def main(path, train_path=True, train_full=False):
    """
    Takes in paths to predictions on test, train, dev set. 

    path: String to directory with all csvs --- Must include / at end
    train_path: Whether train_path is included (or .e.g only predictions on dev & test /// Only on train & test)
    test_path: String
    train_path: String
    dev_path: String
    """

    # Ground truth DFs
    gt_path = "data/"
    train_df = pd.read_json(gt_path + 'train.jsonl', lines=True)
    dev_df = pd.read_json(gt_path + 'devseen.jsonl', lines=True)
    test_df = pd.read_json(gt_path + 'test.jsonl', lines=True)
    test_unseen_df = pd.read_json(gt_path + 'test_unseen.jsonl', lines=True)

    trdv_df = pd.concat([train_df, dev_df], axis=0)

    # Make sure the lists will be ordered, i.e. train[0] is the same model as devs[0]
    train, dev, test, test_unseen = [], [], [], []
    train_probas, dev_probas, test_probas, test_unseen_probas = {}, {}, {}, {} # Never dynamically add to ad pd Dataframe

    for csv in sorted(os.listdir(path)):
        print(csv)
        if ".csv" in csv:
            if "train" in csv:
                train.append(pd.read_csv(path + csv))
                train_probas[csv[:-10]] = pd.read_csv(path + csv).proba.values
            elif ("dev" in csv) or ("val" in csv):
                dev.append(pd.read_csv(path + csv))
                dev_probas[csv[:-8]] = pd.read_csv(path + csv).proba.values
            elif "test_unseen" in csv:
                test_unseen.append(pd.read_csv(path + csv))
                test_unseen_probas[csv[:-14]] = pd.read_csv(path + csv).proba.values
            elif "test" in csv:
                test.append(pd.read_csv(path + csv))
                test_probas[csv[:-7]] = pd.read_csv(path + csv).proba.values

    if train_path or train_full:
        train_probas = pd.DataFrame(train_probas)
    dev_probas = pd.DataFrame(dev_probas)
    test_probas = pd.DataFrame(test_probas)
    test_unseen_probas = pd.DataFrame(test_unseen_probas)

    # Create a combination of tr & dv:
    if train_path:
        traindev = [pd.concat([train[i], dev[i]], axis=0) for i in range(len(train))]
        traindev_probas = pd.concat([train_probas, dev_probas], axis=0)
    # Alternative to train_path, when train is available to fight overfitting to dev
    elif train_full:
        dev = [pd.concat([train[i], dev[i]], axis=0) for i in range(len(train))]
        dev_probas = pd.concat([train_probas, dev_probas], axis=0)
        dev_df = pd.concat([train_df, dev_df], axis=0)


    # Ensembling - Assuming we have train, dev & test
    # STATIC Variables:
    # trdv_df; dev_df, test_df
    # DYNAMIC Variables: 
    # traindev, dev, test
    # traindev_probas, dev_probas, test_probas

    dev_or = dev.copy()
    test_or = test.copy()
    test_unseen_or = test_unseen.copy()

    # As this type of ensembling is based on mostly averages and only 1/5 NN it's very hard to overfit
    loop, last_score, delta = 0, 0, 0.1

    if train_path:
        while (delta > 0.0001):

            # Individual Roc Aucs
            print("Individual RCs:\n")
            print("dev")
            for i, column in enumerate(dev_probas):
                score = roc_auc_score(dev_df.label, dev_probas.iloc[:, i])
                print(column, score)
            print("full")
            for i, column in enumerate(traindev_probas):
                score = roc_auc_score(trdv_df.label, traindev_probas.iloc[:, i])
                print(column, score)
            print('-'*50)

            # Spearman Correlations: 
            print("Spearman Corrs:")
            train_corr = train_probas.corr(method='spearman')
            dev_corr = dev_probas.corr(method='spearman')
            test_corr = test_probas.corr(method='spearman')
            
            print(train_corr, '\n')
            print(dev_corr,'\n')
            print(test_corr)
            print('-'*50)

            ### SIMPLE AVERAGE ###
            trdv_SA = simple_average(traindev_probas, traindev[0], power=1, normalize=True)
            dev_SA = simple_average(dev_probas, dev[0], power=1, normalize=True)
            test_SA = simple_average(test_probas, test[0], power=1, normalize=True)

            print(roc_auc_score(trdv_df.label, trdv_SA.proba), accuracy_score(trdv_df.label, trdv_SA.label))
            print(roc_auc_score(dev_df.label, dev_SA.proba), accuracy_score(dev_df.label, dev_SA.label))
            print('-'*50)

            ### POWER AVERAGE ###
            trdv_PA = simple_average(traindev_probas, traindev[0], power=2, normalize=True)
            dev_PA = simple_average(dev_probas, dev[0], power=2, normalize=True)
            test_PA = simple_average(test_probas, test[0], power=2, normalize=True)

            print(roc_auc_score(trdv_df.label, trdv_PA.proba), accuracy_score(trdv_df.label, trdv_PA.label))
            print(roc_auc_score(dev_df.label, dev_PA.proba), accuracy_score(dev_df.label, dev_PA.label))
            print('-'*50)

            ### RANK AVERAGE ###
            trdv_RA = rank_average(traindev)
            dev_RA = rank_average(dev)
            test_RA = rank_average(test)

            print(roc_auc_score(trdv_df.label, trdv_RA.proba), accuracy_score(trdv_df.label, trdv_RA.label))
            print(roc_auc_score(dev_df.label, dev_RA.proba), accuracy_score(dev_df.label, dev_RA.label))
            print('-'*50)

            ### SIMPLEX ###
            weights_dev = Simplex(dev_probas, dev_df.label)

            dev_SX = simple_average(dev_probas, dev[0], weights_dev)
            test_SX = simple_average(test_probas, test[0], weights_dev)
            dev_trdv_SX = simple_average(traindev_probas, traindev[0], weights_dev)

            weights_trdv = Simplex(traindev_probas, trdv_df.label)

            trdv_dev_SX = simple_average(dev_probas, dev[0], weights_trdv)
            trdv_SX = simple_average(traindev_probas, traindev[0], weights_trdv)
            trdv_test_SX = simple_average(test_probas, test[0], weights_trdv)

            print(roc_auc_score(dev_df.label, trdv_dev_SX.proba), accuracy_score(dev_df.label, trdv_dev_SX.label))
            print(roc_auc_score(dev_df.label, dev_SX.proba), accuracy_score(dev_df.label, dev_SX.label))
            print('-'*50)

            ### NEURAL NETWORK ###
            model = GeluNN(len(dev))
            model = nn_train(model, dev_probas, dev_df.label.values, seed=loop, quiet=True, epochs=50, lr=5e-04)

            dev_NN = nn_predict(dev_probas, dev, model=model)
            test_NN = nn_predict(test_probas, test, model=model)
            dev_trdv_NN = nn_predict(traindev_probas, traindev, model=model)

            print(roc_auc_score(dev_df.label, dev_NN.proba), accuracy_score(dev_df.label, dev_NN.label))

            model = GeluNN(len(dev))
            model = nn_train(model, traindev_probas, trdv_df.label.values, quiet=True, seed=loop, epochs=5, lr=5e-04)

            trdv_dev_NN = nn_predict(dev_probas, dev, model=model)
            trdv_NN = nn_predict(traindev_probas, traindev, model=model)
            trdv_test_NN = nn_predict(test_probas, test, model=model)

            print(roc_auc_score(dev_df.label, trdv_dev_NN.proba), accuracy_score(dev_df.label, trdv_dev_NN.label))
            print('-'*50)

            ############ TESTT
            traindev = [trdv_SA, trdv_PA, trdv_RA, dev_trdv_SX, trdv_SX, dev_trdv_NN, trdv_NN]

            dev = [dev_SA, dev_PA, dev_RA, dev_SX, trdv_dev_SX, dev_NN, trdv_dev_NN]
            test = [test_SA, test_PA, test_RA, test_SX, trdv_test_SX, test_NN, trdv_test_NN]

            

            traindev_probas = pd.concat([df.proba for df in traindev], axis=1)
            dev_probas = pd.concat([df.proba for df in dev], axis=1)
            test_probas = pd.concat([df.proba for df in test], axis=1)

            # Calculate Delta & increment loop
            delta = abs(roc_auc_score(dev_df.label, dev_SX.proba) - last_score)
            last_score = roc_auc_score(dev_df.label, dev_SX.proba)

            loop += 1
    else:

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

            #print("DEV:", dev)
            #print("DEV_PROBAS:", dev_probas)

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
            ### EXP - SIMPLEX WITH RANK AVERAGE
            #dev_SX = rank_average(dev, weights_dev)
            #test_SX = rank_average(test, weights_dev)

            print(roc_auc_score(dev_df.label, dev_SX.proba), accuracy_score(dev_df.label, dev_SX.label))
            print('-'*50)

            ### NEURAL NETWORK ###
            model = GeluNN(len(dev))
            model = nn_train(model, dev_probas, dev_df.label.values, seed=loop, quiet=True, epochs=50, lr=5e-04)

            dev_NN = nn_predict(dev_probas, dev, model=model)
            test_NN = nn_predict(test_probas, test, model=model)
            test_unseen_NN = nn_predict(test_unseen_probas, test_unseen, model=model)

            print(roc_auc_score(dev_df.label, dev_NN.proba), accuracy_score(dev_df.label, dev_NN.label))
            
            dev = dev_or + [dev_SA, dev_PA, dev_RA, dev_SX, dev_NN]
            test = test_or + [test_SA, test_PA, test_RA, test_SX, test_NN]
            test_unseen = test_unseen_or + [test_unseen_SA, test_unseen_PA, test_unseen_RA, test_unseen_SX, test_unseen_NN]
            

            dev_probas = pd.concat([df.proba for df in dev], axis=1)
            test_probas = pd.concat([df.proba for df in test], axis=1)
            test_unseen_probas = pd.concat([df.proba for df in test_unseen], axis=1)

            # Calculate Delta & increment loop
            delta = abs(roc_auc_score(dev_df.label, dev_SX.proba) - last_score)
            last_score = roc_auc_score(dev_df.label, dev_SX.proba)

            loop += 1

            if loop == 2:
                break


    print("Finished with {} after {} loops.".format(last_score, loop))
    # As Simplex at some point simply weighs the highest of all - lets take sx as the final prediction after x loops
    # EXP - any other better?
    #test_unseen_SA.to_csv("/kaggle/working/ens_test_unseen_SA" + args.exp + "_" + str(loop) + ".csv", index=False)
    #test_SA.to_csv("/kaggle/working/ens_test_SA" + args.exp + "_" + str(loop) + ".csv", index=False)
    #dev_SA.to_csv("/kaggle/working/ens_dev_SA" + args.exp + "_" + str(loop) + ".csv", index=False)
    #test_unseen_RA.to_csv("/kaggle/working/ens_test_unseen_RA" + args.exp + "_" + str(loop) + ".csv", index=False)
    #test_RA.to_csv("/kaggle/working/ens_test_RA" + args.exp + "_" + str(loop) + ".csv", index=False)
    #dev_RA.to_csv("/kaggle/working/ens_dev_RA" + args.exp + "_" + str(loop) + ".csv", index=False)
    #test_unseen_PA.to_csv("/kaggle/working/ens_test_unseen_PA" + args.exp + "_" + str(loop) + ".csv", index=False)
    #test_PA.to_csv("/kaggle/working/ens_test_PA" + args.exp + "_" + str(loop) + ".csv", index=False)
    #dev_PA.to_csv("/kaggle/working/ens_dev_PA" + args.exp + "_" + str(loop) + ".csv", index=False)
    #test_unseen_NN.to_csv("/kaggle/working/ens_test_unseen_NN" + args.exp + "_" + str(loop) + ".csv", index=False)
    #test_NN.to_csv("/kaggle/working/ens_test_NN" + args.exp + "_" + str(loop) + ".csv", index=False)
    #dev_NN.to_csv("/kaggle/working/ens_dev_NN" + args.exp + "_" + str(loop) + ".csv", index=False)

    # Get accuracy thresholds & optimize (This does not add value to the roc auc, but just to also have a acc score)
    fpr, tpr, thresholds = metrics.roc_curve(dev_df.label, dev_SX.proba)
    acc, threshold = get_acc_and_best_threshold_from_roc_curve(tpr, fpr, thresholds, 250, 250)
    test_SX.label = test_SX.apply(set_acc, axis=1, args=[threshold])
    test_unseen_SX.label = test_unseen_SX.apply(set_acc, axis=1, args=[threshold])

    test_unseen_SX.to_csv("/kaggle/working/ens_test_unseen" + args.exp + "_" + str(loop) + ".csv", index=False)
    test_SX.to_csv("/kaggle/working/ens_test_" + args.exp + "_" + str(loop) + ".csv", index=False)
    dev_SX.to_csv("/kaggle/working/ens_dev_" + args.exp + "_" + str(loop) + ".csv", index=False)

if __name__ == "__main__":
    
    main(args.enspath, train_path=False, train_full=False)
