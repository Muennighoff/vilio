from collections import defaultdict
import random

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from collections import Counter

from param import args

from sklearn.metrics import roc_auc_score

from fts_tsv.hm_data_tsv import load_obj_tsv

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


class LXMERTDataset(Dataset):
    def __init__(self, splits="train", qa_sets=None):
        super().__init__()
        self.name = splits
        self.splits = splits.split(",")

def make_uid(img_id, dset, sent_idx):
    return "%s_%s_%03d" % (img_id, dset, sent_idx),


"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class LXMERTTorchDataset(Dataset):
    def __init__(self, splits="train",  topk=-1):
        super().__init__()
        self.name = splits
        self.splits = splits.split(",")

        self.task_matched = args.task_matched

        # Loading datasets to data
        self.raw_data = []
        for split in self.splits:
            path = os.path.join("data/", f"{split}.jsonl")
            self.raw_data.extend(
                    [json.loads(jline) for jline in open(path, "r").read().split('\n')]
            )
        print("Load %d data from split(s) %s." % (len(self.raw_data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {datum["id"]: datum for datum in self.raw_data}

        img_data = []

        path = "data/HM_img.tsv"
        img_data.extend(load_obj_tsv(path, self.id2datum.keys()))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            # Adding int here to convert 0625 to 625
            self.imgid2img[int(img_datum['img_id'])] = img_datum

        # Only keep the data with loaded image features
        self.data = []
        for datum in self.raw_data:
            # In HM the Img Id field is simply "id"
            if datum['id'] in self.imgid2img:
                self.data.append(datum)

        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def random_feat(self):
        """Get a random obj feat from the dataset."""
        datum = self.data[random.randint(0, len(self.data)-1)]
        img_id = datum['id']
        img_info = self.imgid2img[img_id]
        feat = img_info['features'][random.randint(0, 35)]
        return feat

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['id']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        obj_labels = img_info['objects_id'].copy()
        obj_confs = img_info['objects_conf'].copy()
        attr_labels = img_info['attrs_id'].copy()
        attr_confs = img_info['attrs_conf'].copy()
        assert obj_num == len(boxes) == len(feats)

        sent = datum['text']

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # If calculating the matched loss, replace the sentence with an sentence
        # corresponding to other image.
        is_matched = 1
        
        if self.task_matched:
            if random.random() < 0.5:
                is_matched = 0
                other_datum = self.data[random.randint(0, len(self.data)-1)]
                while other_datum['id'] == img_id:
                    other_datum = self.data[random.randint(0, len(self.data)-1)]
                sent = other_datum['text']

        label = None

        # Create target
        example = InputExample(
            img_id, sent, (feats, boxes),
            (obj_labels, obj_confs), (attr_labels, attr_confs),
            is_matched, label
        )
        return example