# coding=utf-8
# Copyleft 2019 project LXRT.

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


class HMDataset(Dataset):
    def __init__(self, splits):
        super().__init__()
        self.name = splits
        self.splits = splits.split(",")


class HMTorchDataset(Dataset):
    def __init__(self, splits):
        super().__init__()
        self.name = splits
        self.splits = splits.split(",")

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

        # No idea why, but for hmdatafinal oneimage gets extracted twice causing an error (ID: 81054)
        for o in os.listdir(path):
            if "(1)" in o:
                os.remove(path + o) 

        self.id2file = {int(o.split("_")[0].split(".")[0]): o for o in os.listdir(path)}

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]
        text = datum["text"]
        iid = int(str(datum["id"]).split(".")[0].split("_")[0])

        img = self.process_img(iid)

        # Get image info
        img_h = img["img_h"]
        img_w = img["img_w"]
        feats = torch.FloatTensor(img["feats"][:100, ...]).clone()
        boxes = torch.FloatTensor(img["pos"][:100, ...]).clone()
        pred_objs = img["pred_objs"]
        pred_conf = img["pred_conf"]
        assert len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        #boxes = boxes.clone()
        #boxes[:, (0, 2)] /= img_w
        #boxes[:, (1, 3)] /= img_h
        #np.testing.assert_array_less(boxes, 1 + 1e-5)
        #np.testing.assert_array_less(-boxes, 0 + 1e-5)

        # Create target
        if "label" in datum:
            target = torch.tensor(datum["label"], dtype=torch.float) 
            return iid, feats, boxes, text, target
        else:
            return iid, feats, boxes, text

class HMEvaluator:
    def __init__(self, dataset):
        self.dataset = dataset

    def evaluate(self, id2ans: dict):
        score = 0.0
        total = 0.0
        for img_id, ans in id2ans.items():
            datum = self.dataset.id2datum[int(img_id)]
            label = datum["label"]
            if ans == label:
                score += 1
            total += 1
 
        return score / total

    def dump_json(self, id2ans: dict, path):

        with open(path, "w") as f:
            result = []
            for img_id, ans in id2ans.items():
                result.append({"img_id": img_id, "pred": ans})
            json.dump(result, f, indent=4, sort_keys=True)

    def dump_csv(self, id2ans: dict, id2prob: dict, path):

        d = {"id": [int(tensor) for tensor in id2ans.keys()], "proba": list(id2prob.values()), 
            "label": list(id2ans.values())}
        results = pd.DataFrame(data=d)
        
        print(results.info())

        results.to_csv(path_or_buf=path, index=False)

    def roc_auc(self, id2ans:dict):
        """Calculates roc_auc score"""
        ans = list(id2ans.values())
        label = [self.dataset.id2datum[int(key)]["label"] for key in id2ans.keys()]
        score = roc_auc_score(label, ans)
        return score