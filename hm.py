import collections
import os

from param import args

import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.nn.modules.loss import _Loss

if args.tsv:
    from hm_data_tsv import HMTorchDataset, HMEvaluator, HMDataset
else:
    from hm_data import HMTorchDataset, HMEvaluator, HMDataset

from src.vilio.transformers.optimization import AdamW, get_linear_schedule_with_warmup

from entryU import ModelU
from entryX import ModelX
from entryV import ModelV
from entryD import ModelD
from entryO import ModelO

# Two different SWA Methods - https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
if args.swa:
    from torch.optim.swa_utils import AveragedModel, SWALR
    from torch.optim.lr_scheduler import CosineAnnealingLR

if args.contrib:
    from torchcontrib.optim import SWA


# Largely sticking to the standards set in LXMERT here
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:

    dset =  HMDataset(splits)

    tset = HMTorchDataset(splits)
    evaluator = HMEvaluator(tset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)

class RocStarLoss(_Loss):
    """Smooth approximation for ROC AUC
    """
    def __init__(self, delta = 1.0, sample_size = 10, sample_size_gamma = 10, update_gamma_each=10):
        r"""
        Args:
            delta: Param from article
            sample_size (int): Number of examples to take for ROC AUC approximation
            sample_size_gamma (int): Number of examples to take for Gamma parameter approximation
            update_gamma_each (int): Number of steps after which to recompute gamma value.
        """
        super().__init__()
        self.delta = delta
        self.sample_size = sample_size
        self.sample_size_gamma = sample_size_gamma
        self.update_gamma_each = update_gamma_each
        self.steps = 0
        size = max(sample_size, sample_size_gamma)

        # Randomly init labels
        self.y_pred_history = torch.rand((size, 1))
        self.y_true_history = torch.randint(2, (size, 1))
        

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Tensor of model predictions in [0, 1] range. Shape (B x 1)
            y_true: Tensor of true labels in {0, 1}. Shape (B x 1)
        """
        #y_pred = _y_pred.clone().detach()
        #y_true = _y_true.clone().detach()
        if self.steps % self.update_gamma_each == 0:
            self.update_gamma()
        self.steps += 1
        
        positive = y_pred[y_true > 0]
        negative = y_pred[y_true < 1]
        
        # Take last `sample_size` elements from history
        y_pred_history = self.y_pred_history[- self.sample_size:]
        y_true_history = self.y_true_history[- self.sample_size:]
        
        positive_history = y_pred_history[y_true_history > 0]
        negative_history = y_pred_history[y_true_history < 1]
        
        if positive.size(0) > 0:
            diff = negative_history.view(1, -1) + self.gamma - positive.view(-1, 1)
            loss_positive = torch.nn.functional.relu(diff ** 2).mean()
        else:
            loss_positive = 0
 
        if negative.size(0) > 0:
            diff = negative.view(1, -1) + self.gamma - positive_history.view(-1, 1)
            loss_negative = torch.nn.functional.relu(diff ** 2).mean()
        else:
            loss_negative = 0
            
        loss = loss_negative + loss_positive
        
        # Update FIFO queue
        batch_size = y_pred.size(0)
        self.y_pred_history = torch.cat((self.y_pred_history[batch_size:], y_pred.clone().detach()))
        self.y_true_history = torch.cat((self.y_true_history[batch_size:], y_true.clone().detach()))
        return loss

    def update_gamma(self):
        # Take last `sample_size_gamma` elements from history
        y_pred = self.y_pred_history[- self.sample_size_gamma:]
        y_true = self.y_true_history[- self.sample_size_gamma:]
        
        positive = y_pred[y_true > 0]
        negative = y_pred[y_true < 1]
        
        # Create matrix of size sample_size_gamma x sample_size_gamma
        diff = positive.view(-1, 1) - negative.view(1, -1)
        AUC = (diff > 0).type(torch.float).mean()
        num_wrong_ordered = (1 - AUC) * diff.flatten().size(0)
        
        # Adjuct gamma, so that among correct ordered samples `delta * num_wrong_ordered` were considered
        # ordered incorrectly with gamma added
        correct_ordered = diff[diff > 0].flatten().sort().values
        idx = min(int(num_wrong_ordered * self.delta), len(correct_ordered)-1)
        self.gamma = correct_ordered[idx]

class HM:
    def __init__(self):
        
        if args.train is not None:
            self.train_tuple = get_tuple(
                args.train, bs=args.batch_size, shuffle=True, drop_last=False
            )

        if args.valid is not None:
            valid_bsize = 2048 if args.multiGPU else 50
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        # Select Model, X is default
        if args.model == "X":
            self.model = ModelX(args)
        elif args.model == "V":
            self.model = ModelV(args)
        elif args.model == "U":
            self.model = ModelU(args)
        elif args.model == "D":
            self.model = ModelD(args)
        elif args.model == 'O':
            self.model = ModelO(args)
        else:
            print(args.model, " is not implemented.")

        # Load pre-trained weights from paths
        if args.loadpre is not None:
            self.model.load(args.loadpre)

        # GPU options
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        self.model = self.model.cuda()

        # Losses and optimizer
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.nllloss = nn.NLLLoss()

        self.roc_star_loss = RocStarLoss()

        if args.train is not None:
            batch_per_epoch = len(self.train_tuple.loader)
            self.t_total = int(batch_per_epoch * args.epochs // args.acc)
            print("Total Iters: %d" % self.t_total)

        def is_backbone(n):
            if "encoder" in n:
                return True
            elif "embeddings" in n:
                return True
            elif "pooler" in n:
                return True
            print("F: ", n)
            return False

        no_decay = ['bias', 'LayerNorm.weight']

        params = list(self.model.named_parameters())
        if args.reg:
            optimizer_grouped_parameters = [
                {"params": [p for n, p in params if is_backbone(n)], "lr": args.lr},
                {"params": [p for n, p in params if not is_backbone(n)], "lr": args.lr * 500},
            ]

            for n, p in self.model.named_parameters():
                print(n)

            self.optim = AdamW(optimizer_grouped_parameters, lr=args.lr)
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
                {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]

            self.optim = AdamW(optimizer_grouped_parameters, lr=args.lr)

        if args.train is not None:
            self.scheduler = get_linear_schedule_with_warmup(self.optim, self.t_total * 0.1, self.t_total)
        
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

        if args.contrib:
            self.optim = SWA(self.optim, swa_start=self.t_total * 0.75, swa_freq=5, swa_lr=args.lr)

        if args.swa: 
            self.swa_model = AveragedModel(self.model)
            self.swa_start = self.t_total * 0.75
            self.swa_scheduler = SWALR(self.optim, swa_lr=args.lr)

    def train(self, train_tuple, eval_tuple):

        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        print("Batches:", len(loader))

        self.optim.zero_grad()

        best_roc = 0.
        ups = 0
        

        total_loss = 0.

        for epoch in range(args.epochs):
            
            if args.reg:
                if args.model != "X":
                    print(self.model.model.layer_weights)

            quesid2ans = {}
            quesid2prob = {}

            #epoch_y_pred = []
            #epoch_y_t = []

            for i, (ques_id, feats, boxes, sent, label, target) in iter_wrapper(enumerate(loader)):

                if ups == args.midsave:
                    self.save("MID")

                self.model.train()

                if args.swa:
                    self.swa_model.train()
                
                target = target.long().cuda()

                # Model expects visual feats as tuple of feats & boxes
                logit = self.model(sent, (feats, boxes))

                # Note: LogSoftmax does not change order, hence there should be nothing wrong with taking it as our prediction 
                # In fact ROC AUC stays the exact same for logsoftmax / normal softmax, but logsoftmax is better for loss calculation
                # due to stronger penalization & decomplexifying properties (log(a/b) = log(a) - log(b))
                logit = self.logsoftmax(logit)
                score = logit[:, 1]

                if i < 1:
                    print(logit[0, :].detach())
               

                if (epoch > 0) and (args.rcac):
                    loss = self.roc_star_loss(score, target)
                else:
                    # Note: This loss is the same as CrossEntropy (We splitted it up in logsoftmax & neg. log likelihood loss)
                    loss = self.nllloss(logit.view(-1, 2), target.view(-1))

                # Scaling loss by batch size, as we have batches with different sizes, since we do not "drop_last" & dividing by acc for accumulation
                # Not scaling the loss will worsen performance by ~2abs%
                loss = loss * logit.size(0) / args.acc
                loss.backward()

                total_loss += loss.detach().item()

                # Acts as argmax - extracting the higher score & the corresponding index (0 or 1)
                _, predict = logit.detach().max(1)
                # Getting labels for accuracy
                for qid, l in zip(ques_id, predict.cpu().numpy()):
                    quesid2ans[qid] = l
                # Getting probabilities for Roc auc
                for qid, l in zip(ques_id, score.detach().cpu().numpy()):
                    quesid2prob[qid] = l


                if (i+1) % args.acc == 0:

                    nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)

                    self.optim.step()

                    if (args.swa) and (ups > self.swa_start):
                        self.swa_model.update_parameters(self.model)
                        self.swa_scheduler.step()
                    else:
                        self.scheduler.step()
                    self.optim.zero_grad()

                    ups += 1

                    # Do Validation in between
                    if ups % 250 == 0: 
                        
                        log_str = "\nEpoch(U) %d(%d): Train AC %0.2f RA %0.4f LOSS %0.4f\n" % (epoch, ups, evaluator.evaluate(quesid2ans)*100, 
                        evaluator.roc_auc(quesid2prob)*100, total_loss)

                        # Set loss back to 0 after printing it
                        total_loss = 0.

                        if self.valid_tuple is not None:  # Do Validation
                            acc, roc_auc, roc_auc0, roc_auc1 = self.evaluate(eval_tuple)
                            if roc_auc > best_roc:
                                best_roc = roc_auc
                                best_acc = acc
                                # Not saving Best for now as we only use LAST
                                # self.save("BEST")

                            log_str += "\nEpoch(U) %d(%d): DEV AC %0.2f RA %0.4f RA0 %0.4f RA1 %0.4f\n" % (epoch, ups, acc*100.,roc_auc*100, roc_auc0*100, 
                            roc_auc1*100)
                            log_str += "Epoch(U) %d(%d): Best (RA) ACC %0.2f RA %0.4f \n" % (epoch, ups, best_acc*100., best_roc*100.)
    
                        print(log_str, end='')

                        with open(self.output + "/log.log", 'a') as f:
                            f.write(log_str)
                            f.flush()

        if (epoch + 1) == args.epochs:
            if args.contrib:
                self.optim.swap_swa_sgd()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None, out_csv=True):

        dset, loader, evaluator = eval_tuple
        quesid2ans = {}

        quesid2prob = {}
        quesid2prob0 = {}
        quesid2prob1 = {}

        for i, datum_tuple in enumerate(loader):

            ques_id, feats, boxes, sent = datum_tuple[:4]

            self.model.eval()

            if args.swa:
                self.swa_model.eval()

            with torch.no_grad():
                
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(sent, (feats, boxes))

                # Taking several raw estimates
                score0 = logit[:, 1]
                score1 = 100 - logit[:, 0] # Inverting; 100 needed as - & + values

                # Note: LogSoftmax does not change order, hence there should be nothing wrong with taking it as our prediction
                logit = self.logsoftmax(logit)
                score = logit[:, 1]

                if args.swa:
                    logit_swa = self.swa_model(sent, (feats, boxes))
                    logit_swa = self.logsoftmax(logit_swa)
                    score1 = logit_swa[:, 1]

                _, predict = logit.max(1)

                for qid, l in zip(ques_id, predict.cpu().numpy()):
                    quesid2ans[qid] = l

                # Getting probas for Roc Auc
                for qid, l in zip(ques_id, score.cpu().numpy()):
                    quesid2prob[qid] = l

                for qid, l in zip(ques_id, score0.cpu().numpy()):
                    quesid2prob0[qid] = l
                
                for qid, l in zip(ques_id, score1.cpu().numpy()):
                    quesid2prob1[qid] = l


        if dump is not None:
            if out_csv == True:
                evaluator.dump_csv(quesid2ans, quesid2prob, dump)
            else:
                evaluator.dump_result(quesid2ans, dump)

        return quesid2ans, quesid2prob, quesid2prob0, quesid2prob1

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans, quesid2prob, quesid2prob0, quesid2prob1 = self.predict(eval_tuple, dump=dump)
        acc = eval_tuple.evaluator.evaluate(quesid2ans)

        roc_auc = eval_tuple.evaluator.roc_auc(quesid2prob)
        roc_auc0 = eval_tuple.evaluator.roc_auc(quesid2prob0)
        roc_auc1 = eval_tuple.evaluator.roc_auc(quesid2prob1)

        return acc, roc_auc, roc_auc0, roc_auc1

    def save(self, name):
        if args.swa:
            torch.save(self.swa_model.state_dict(),
                    os.path.join(self.output, "%s.pth" % name))
        else:
            torch.save(self.model.state_dict(),
                    os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
            
        state_dict = torch.load("%s" % path)
        new_state_dict = {}
        for key, value in state_dict.items():
            # N_averaged is a key in SWA models we cannot load, so we skip it
            if key.startswith("n_averaged"):
                print("n_averaged:", value)
                continue
            # SWA Models will start with module
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
        self.model.load_state_dict(state_dict)

if __name__ == "__main__":

    # Build Class
    hm = HM()

    # Load Model
    if args.loadfin is not None:
        hm.load(args.loadfin)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        # To avoid having to reload the tsv everytime:
        for split in args.test.split(","):
            if 'test' in split:
                hm.predict(
                    get_tuple(split, bs=args.batch_size,
                            shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, '{}_{}.csv'.format(args.exp, split))
                )
            elif 'dev' in split or 'valid' in split or 'train' in split:
                result = hm.evaluate(
                    get_tuple(split, bs=args.batch_size,
                            shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, '{}_{}.csv'.format(args.exp, split))
                )
                print(result)
            else:
                assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', hm.train_tuple.dataset.splits)
        if hm.valid_tuple is not None:
            print('Splits in Valid data:', hm.valid_tuple.dataset.splits)
        else:
            print("DO NOT USE VALIDATION")
        hm.train(hm.train_tuple, hm.valid_tuple)

