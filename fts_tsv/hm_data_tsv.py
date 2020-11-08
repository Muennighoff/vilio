# Mix of hm_data & LXMERTs task data files

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
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
        self.raw_data = []
        for split in self.splits:
            path = os.path.join("data/", f"{split}.jsonl")
            self.raw_data.extend(
                    [json.loads(jline) for jline in open(path, "r").read().split('\n')]
            )
        print("Load %d data from split(s) %s." % (len(self.raw_data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {datum["id"]: datum for datum in self.raw_data}

        # Loading detection features to img_data
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


    def __getitem__(self, item: int):

        datum = self.data[item]

        img_id = datum['id']
        text = datum['text']


        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)


        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']

        if args.num_pos == 5: 
            # For DeVLBert taken from VilBERT
            image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
            image_location[:,:4] = boxes
            image_location[:,4] = (image_location[:,3] - image_location[:,1]) * (image_location[:,2] - image_location[:,0]) / (float(img_w) * float(img_h))
            boxes = image_location

        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)


        if args.num_pos == 6:
            # Add width & height
            width = (boxes[:, 2] - boxes[:, 0]).reshape(-1,1)
            height = (boxes[:, 3] - boxes[:, 1]).reshape(-1,1)

            boxes = np.concatenate((boxes, width, height), axis=-1)

            # In UNITER they use 7 Pos Feats (See _get_img_feat function in their repo)
            if args.model == "U":
                boxes = np.concatenate([boxes, boxes[:, 4:5]*boxes[:, 5:]], axis=-1)

        # Provide label (target) - From hm_data
        if 'label' in datum:
            target = torch.tensor(datum["label"], dtype=torch.float) 
            return img_id, feats, boxes, text, target
        else:
            return img_id, feats, boxes, text

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


### TSV EXTRACTION

import sys
import csv
import base64
import time

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

def load_obj_tsv(fname, ids, topk=args.topk):
    """Load object features from tsv file.
    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        boxes = args.num_features # Same boxes for all

        for i, item in enumerate(reader):
            
            # Check if id in list of ids to save memory
            if int(item["img_id"]) not in ids:
                continue

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                try:
                    item[key] = item[key].reshape(shape)
                except:
                    # In 1 out of 10K cases, the shape comes out wrong; We make necessary adjustments
                    shape = list(shape)
                    shape[0] += 1
                    shape = tuple(shape)
                    item[key] = item[key].reshape(shape)  
 
                item[key].setflags(write=False)
                    
            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data