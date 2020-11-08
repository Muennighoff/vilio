import argparse
import random

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    # Data Splits
    parser.add_argument("--train", default=None)
    parser.add_argument("--valid", default=None)
    parser.add_argument("--test", default=None)

    # Data-Type
    parser.add_argument("--tsv", action='store_const', default=False, const=True, help='Whether to use tsv extraction (else lmdb)')
    parser.add_argument("--num_features", type=int, default=100, help='How many features we have per img (e.g. 100, 80)')
    parser.add_argument("--num_pos", type=int, default=4, help='How many position feats - 4 or 6')

    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=8)
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument("--clip", type=float, default=5.0, help='Clip Grad Norm')
    parser.add_argument("--wd", type=float, default=0.0, help='Weight Decay')
    parser.add_argument("--swa", action='store_const', default=False, const=True)
    parser.add_argument("--contrib", action='store_const', default=False, const=True)
    parser.add_argument("--case", action='store_const', default=False, const=True)
    parser.add_argument("--reg", action='store_const', default=False, const=True) # Applies Multi-sample dropout & Layeravg
    parser.add_argument("--acc", type=int, default=1, help='Amount of accumulation steps for bigger batch size - make sure to adjust LR')
    parser.add_argument('--subtrain', action='store_const', default=False, const=True)
    parser.add_argument('--subtest', action='store_const', default=False, const=True)
    parser.add_argument('--combine', action='store_const', default=False, const=True, help="Combine subtrained data; subtrain must be true")

    # Debugging
    parser.add_argument('--output', type=str, default='./data', help='Where ckpts, csvs shall be saved to')
    parser.add_argument("--tqdm", action='store_const', default=False, const=True)
    parser.add_argument("--topk", type=int, default=-1, help='For testing only load topk feats from tsv')
    parser.add_argument("--exp", type=str, default="experiment", help="Name of experiment for csv's")

    # Model Loading & Saving - Note: PATHS must be put in here! 
    parser.add_argument('--model', type=str, default='X', help='Type of Model, X V O U D')
    parser.add_argument("--tr", type=str, default="bert-base-uncased", help="Name of NLP transformer to be used")
    parser.add_argument("--midsave", type=int, default=-1, help='Save a MID model after x steps')
    parser.add_argument('--loadfin', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')               
    parser.add_argument('--loadpre', type=str, default=None,
                        help='Load the pre-trained model.')
    parser.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True,
                        help='If none of the --load, --loadLXMERT, --loadLXMERTQA is set, '
                             'the model would be trained from scratch. If --fromScratch is'
                             ' not specified, the model would load BERT-pre-trained weights by'
                             ' default. ')

    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    parser.add_argument("--llayers", default=9, type=int, help='Number of Language layers')
    parser.add_argument("--xlayers", default=5, type=int, help='Number of CROSS-modality layers.')
    parser.add_argument("--rlayers", default=5, type=int, help='Number of object Relationship layers.')

    # Pre-training Config
    parser.add_argument("--taskHM", dest='task_hm', action='store_const', default=False, const=True)
    parser.add_argument("--taskMatched", dest='task_matched', action='store_const', default=False, const=True)
    parser.add_argument("--taskMaskLM", dest='task_mask_lm', action='store_const', default=False, const=True)
    parser.add_argument("--taskObjPredict", dest='task_obj_predict', action='store_const', default=False, const=True)
    parser.add_argument("--taskQA", dest='task_qa', action='store_const', default=False, const=True)
    parser.add_argument("--visualLosses", dest='visual_losses', default='obj,attr,feat', type=str)
    parser.add_argument("--qaSets", dest='qa_sets', default=None, type=str)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15, type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate', default=0.15, type=float)

    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument("--numWorkers", dest='num_workers', default=4)

    # Parse the arguments.
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()