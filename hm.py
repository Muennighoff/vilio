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
    from hm_data_tsv import MMFTorchDataset, MMFEvaluator, MMFDataset
else:
    from hm_data import MMFTorchDataset, MMFEvaluator, MMFDataset

from src.vilio.transformers.optimization import AdamW, get_linear_schedule_with_warmup

from entryU import ModelU
from entryX import ModelX
from entryV import ModelV
from entryD import ModelD
from entryO import ModelO

# SWA:
if args.swa:
    from torch.optim.swa_utils import AveragedModel, SWALR
    from torch.optim.lr_scheduler import CosineAnnealingLR


# Largely sticking to the standards set in LXMERT here
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False, rs=True) -> DataTuple:

    dset =  MMFDataset(splits)

    tset = MMFTorchDataset(splits, rs=rs)
    evaluator = MMFEvaluator(tset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)

class MMF:
    def __init__(self):
        
        if args.train != "":
            self.train_tuple = get_tuple(
                args.train, bs=args.batch_size, shuffle=True, drop_last=False
            )

        if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 50
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False, rs=False
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
        if args.load_lxmert is not None:
            self.model.load(args.load_lxmert)

        # GPU options
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        self.model = self.model.cuda()

        # Losses and optimizer
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.nllloss = nn.NLLLoss()

        if args.train != "":
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

        self.scheduler = get_linear_schedule_with_warmup(self.optim, self.t_total * 0.1, self.t_total)
        
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

        # Implementing SWA as in https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        if args.contrib:
            from torchcontrib.optim import SWA
            self.optim = SWA(self.optim, swa_start=self.t_total * 0.75, swa_freq=5, swa_lr=args.lr)

        if args.swa: 
            self.swa_model = AveragedModel(self.model)
            self.swa_start = self.t_total * 0.75
            self.swa_scheduler = SWALR(self.optim, swa_lr=args.lr)

    def roc_star_loss(self, _y_true, y_pred, gamma, _epoch_true, epoch_pred):
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
        # Convert labels to boolean
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

    def epoch_update_gamma(self, y_true, y_pred, epoch=-1, delta=1.0):
        """
        Calculate gamma from last epoch's targets and predictions.
        Gamma is updated at the end of each epoch.
        y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
        y_pred: `Tensor` . Predictions.
        """
        DELTA = delta
        SUB_SAMPLE_SIZE = 2000.0

        pos = y_pred[y_true==1] 
        neg = y_pred[y_true==0]
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
            gamma = default_gamma # default=torch.tensor(0.2, dtype=torch.float).cuda()
        L1 = diff[diff>-1.0*gamma]
        ln_L1 = L1.shape[0]
        if epoch > -1 :
            return gamma
        else :
            return default_gamma


    def train(self, train_tuple, eval_tuple):

        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        print("Batches:", len(loader))

        self.optim.zero_grad()

        best_roc = 0.
        ups = 0
        
        epoch_gamma = 0.20

        total_loss = 0.

        for epoch in range(args.epochs):
            
            if args.reg:
                if args.model != "X":
                    print(self.model.model.layer_weights)

            quesid2ans = {}
            quesid2prob = {}

            epoch_y_pred = []
            epoch_y_t = []

            for i, (ques_id, feats, boxes, num_b, sent, label, target) in iter_wrapper(enumerate(loader)):

                if ups == args.midsave:
                    self.save("MID")

                self.model.train()

                if args.swa:
                    self.swa_model.train()
                
                target = target.long().cuda()

                # Model expects visual feats as tuple of feats & boxes
                logit = self.model(sent, (feats, boxes))

                # Taking raw estimates (Shifted to val-only) (Removed Softmax as same as logsoftmax)
                #score1 = logit[:, 1]
                #score2 = 100 - logit[:, 0] # Inverting; 100 needed as - & + values

                # Note: LogSoftmax does not change order, hence there should be nothing wrong with taking it as our prediction 
                # In fact ROC AUC stays the exact same for logsoftmax / normal softmax, but logsoftmax is better for loss calculation
                # due to stronger penalization & decomplexifying properties (log(a/b) = log(a) - log(b))
                logit = self.logsoftmax(logit)
                score = logit[:, 1]

                if i < 1:
                    print(logit[0, :].detach())
               
                #target = target.unsqueeze(dim=1).long()#.float()#.cuda()
                #score1 = score1.unsqueeze(dim=1)#.long()#.cuda()
                #score1 = torch.tensor(score1, requires_grad=True, dtype=torch.long)#(requires_grad=True)

                if (epoch > 0) and (args.rcac):
                    loss = self.roc_star_loss(target, score, epoch_gamma, last_epoch_y_t, last_epoch_y_pred)
                else:
                    # Note: This loss is the same as CrossEntropy (We splitted it up in logsoftmax & neg. log likelihood loss)
                    loss = self.nllloss(logit.view(-1, 2), target.view(-1))

                # Scaling loss by batch size, as we have batches with different sizes, since we do not "drop_last" & dividing by acc for accumulation
                # Not scaling the loss will worsen performance by ~2abs%
                loss = loss * logit.size(0) / args.acc
                loss.backward()

                total_loss += loss.detach().item()

                if args.rcac:
                    epoch_y_pred.extend(score.detach())
                    epoch_y_t.extend(target)

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

            if args.rcac:
                last_epoch_y_pred = torch.tensor(epoch_y_pred).cuda()
                last_epoch_y_t = torch.tensor(epoch_y_t).cuda()

                epoch_gamma = self.epoch_update_gamma(last_epoch_y_t, last_epoch_y_pred, epoch)

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

            ques_id, feats, boxes, num_b, sent = datum_tuple[:5]

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
            
        state_dict = torch.load("%s.pth" % path)
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
    mmf = MMF()

    # Load Model
    if args.load is not None:
        mmf.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        # To avoid having to reload the tsv everytime:
        for split in args.test.split(","):
            if 'test' in split:
                mmf.predict(
                    get_tuple(split, bs=args.batch_size,
                            shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, '{}_{}.csv'.format(args.exp, split))
                )
            elif 'dev' in split or 'valid' in split or 'train' in split:
                result = mmf.evaluate(
                    get_tuple(split, bs=args.batch_size,
                            shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, '{}_{}.csv'.format(args.exp, split))
                )
                print(result)
            else:
                assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', mmf.train_tuple.dataset.splits)
        if mmf.valid_tuple is not None:
            print('Splits in Valid data:', mmf.valid_tuple.dataset.splits)
        else:
            print("DO NOT USE VALIDATION")
        mmf.train(mmf.train_tuple, mmf.valid_tuple)

