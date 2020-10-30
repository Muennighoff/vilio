# coding=utf-8
# Copyleft 2019 project LXRT.

from collections import defaultdict
import random

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from fts_lmdb.features_database import FeaturesDatabase
import os
from collections import Counter

from param import args

from sklearn.metrics import roc_auc_score


class InputExample(object):
    """A single training/test example for the language model."""
    def __init__(self, uid, sent, visual_feats=None,
                 obj_labels=None, attr_labels=None,
                 is_matched=None, label=None, vl_label=None):
        self.uid = uid
        self.sent = sent
        self.visual_feats = visual_feats
        self.obj_labels = obj_labels
        self.attr_labels = attr_labels
        self.is_matched = is_matched  # whether the visual and obj matched
        self.label = label
        self.vl_label = vl_label


class LXMERTDataset(Dataset):
    def __init__(self, splits="train", qa_sets=None):
        super().__init__()
        self.name = splits
        self.splits = splits.split(",")

def make_uid(img_id, dset, sent_idx):
    return "%s_%s_%03d" % (img_id, dset, sent_idx),


class LXMERTTorchDataset(Dataset):
    def __init__(self, splits="train",  topk=-1):
        super().__init__()
        self.name = splits
        self.splits = splits.split(",")

        self.task_matched = args.task_matched

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            path = os.path.join("data/", f"{split}.jsonl")
            self.data.extend(
                    [json.loads(jline) for jline in open(path, "r").read().split('\n')]
            )
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {datum["id"]: datum for datum in self.data}
        path = "data/features/"
        path2 = "data/detectron.lmdb"
        self.db = FeaturesDatabase(
                path=path2,
                annotation_db=None,
                feature_path=path)

        # No idea why, but for fbmdata final image gets extracted twice causing an error (ID: 81054)
        for o in os.listdir(path):
            if "(1)" in o:
                os.remove(path + o) 

        self.id2file = {int(o.split("_")[0].split(".")[0]): o for o in os.listdir(path)}

        print("Use %d data in torch dataset" % (len(self.data)))

    def process_img(self, iid):
        f = self.id2file[iid]
        item = self.db.from_path(f)
        return {
            "gt_objs": item["image_info_0"]["objects"],
            "img_h": item["image_info_0"]["image_height"],
            "img_w": item["image_info_0"]["image_width"],
            "feats": item["image_feature_0"],
            "pos": item["image_info_0"]["bbox"],
            "pred_objs": torch.max(torch.FloatTensor(item["image_info_0"]["cls_prob"]), -1).indices,
            "pred_conf": torch.max(torch.FloatTensor(item["image_info_0"]["cls_prob"]), -1).values,
            "cls_prob": item["image_info_0"]["cls_prob"]
        }

    def random_feat(self):
        """
        Get a random obj feat from the dataset.
        Mix of mmf __getitem__ & random_feat.
        """

        datum = self.data[random.randint(0, len(self.data)-1)]
        iid = int(str(datum["id"]).split(".")[0].split("_")[0])
        img = self.process_img(iid)
        feats = torch.FloatTensor(img["feats"][:100, ...]).clone()

        feat = feats[random.randint(0, 99)]

        return feat

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        """
        Mix of __getitem__ in original pre_train and mmf_data
        """
        datum = self.data[item]

        iid = int(str(datum["id"]).split(".")[0].split("_")[0])

        img = self.process_img(iid)

        # Get image info - A lot less than in or repo
        img_h = img["img_h"]
        img_w = img["img_w"]
        feats = torch.FloatTensor(img["feats"][:100, ...]).clone()
        boxes = torch.FloatTensor(img["pos"][:100, ...]).clone()
        pred_objs = img["pred_objs"]
        pred_conf = img["pred_conf"]
        assert len(boxes) == len(feats)
        
        # If calculating the matched loss, replace the sentence with an sentence
        # corresponding to other image.
        is_matched = 1
        sent = datum['text']
        if self.task_matched:
            if random.random() < 0.5:
                is_matched = 0
                other_datum = self.data[random.randint(0, len(self.data)-1)]
                # Keep looking for another datum while same image
                while int(str(other_datum["id"]).split(".")[0].split("_")[0]) == iid:
                    other_datum = self.data[random.randint(0, len(self.data)-1)]

                sent = other_datum['text']
                
        # If label
        if "label" in datum:
            if int(datum["label"]) == 1:
                label = [0, 1]
            else:
                label = [1, 0]
            target = torch.tensor(datum["label"], dtype=torch.float) 
            label = torch.tensor(label, dtype=torch.float)
        else:
            # Set the target to the ignore_index of later used Cross Entropy Loss
            target = torch.tensor([-1], dtype=torch.float)

        vl_label = target

        # Missing for us
        uid = None
        attr_labels = None
        attr_confs = None
        ans_label = None


        # Create target
        example = InputExample(
            uid, sent, (feats, boxes),
            (pred_objs, pred_conf), (attr_labels, attr_confs),
            is_matched, ans_label, vl_label
        )

        return example
        
