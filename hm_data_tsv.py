# Mix of mmf_data & LXMERTs task data files

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from collections import Counter

from param import args

from sklearn.metrics import roc_auc_score

from vg_dict import vg_dict


class MMFDataset(Dataset):
    def __init__(self, splits):
        super().__init__()
        self.name = splits
        self.splits = splits.split(",")

class MMFTorchDataset(Dataset):
    def __init__(self, splits, rs=True):
        super().__init__()
        self.name = splits
        self.splits = splits.split(",")

        self.rs = rs

        # Loading datasets to data
        self.raw_data = []
        for split in self.splits:
            #path = os.path.join(os.pardir, "data/hateful_memes/data", f"{split}.jsonl")
            path = os.path.join("data/", f"{split}.jsonl")
            self.raw_data.extend(
                    [json.loads(jline) for jline in open(path, "r").read().split('\n')]
            )
        print("Load %d data from split(s) %s." % (len(self.raw_data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {datum["id"]: datum for datum in self.raw_data}

        ### ABOVE could be moved to "Dataset" ###


        # Loading detection features to img_data
        img_data = []

        path = "data/HM_img.tsv"
        img_data.extend(load_obj_tsv(path, self.id2datum.keys()))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            # Adding int here to convert 0625 to 625
            self.imgid2img[int(img_datum['img_id'])] = img_datum

        # AUG EXP
        #if self.rs and args.model != "U":
        #    path_gt = "data/HM_gt_img.tsv"
        #    img_data_gt = []#

        #    img_data_gt.extend(load_obj_tsv(path_gt, self.id2datum.keys()))
            # Convert img list to dict
        #    self.imgid2img_gt = {}
        #    for img_datum in img_data_gt:
               # Adding int here to convert 0625 to 625
        #        self.imgid2img_gt[int(img_datum['img_id'])] = img_datum



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

        # Exp. Random Swap augmentation
        #if self.rs == True:
        #    text = random_swap(text, 2)
        #    text = random_insertion(text, int(len(text) * random.uniform(0, 0.1))) #int(len(text) * random.uniform(0, 0.2)))

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Select Objects & Attributes to append to text
        if args.textb:

            obj_id = img_info["objects_id"]
            obj_conf = img_info["objects_conf"]
            attr_id = img_info["attrs_id"]
            attr_conf = img_info["attrs_conf"]

            assert len(obj_id) == len(obj_conf) == len(attr_id) == len(attr_conf)

            textb = "[SEP]"

            counts = Counter()

            for oid, oconf, aid, aconf in zip(obj_id, obj_conf, attr_id, attr_conf):
                attr_obj = ""
                if aconf > 999:
                    for entry in vg_dict["attCategories"]:
                        if entry["id"] == aid:
                            attr_obj += " " + entry["name"]
            
                if oconf > 0.3:
                    for entry in vg_dict["categories"]:
                        if entry["id"] == oid:
                            attr_obj += " " + entry["name"]

                #(TAB THIS FORWARD)  
                # We only add if we have also found an obj for the attr
                counts[attr_obj] += 1
                
                # Only take unique attr_obj combos
                if counts[attr_obj] < 2:
                    textb += attr_obj

                ### Cut short at e.g. 10:
                if len(counts) > 10:
                    break

            # Add it to our normal text
            text += textb

        # Normalize the boxes (to 0 ~ 1)

        # Exp. Random Swap augmentation
        #if self.rs == True and args.model != "U":
        #    boxes, feats = random_swap_feats(boxes, feats, 2)
        #    img_info_gt = self.imgid2img_gt[img_id]
        #    feats_gt = img_info_gt['features'].copy()
        #    boxes_gt = img_info_gt['boxes'].copy()
        #    boxes, feats = random_swap_feats_gt(boxes, feats, boxes_gt, feats_gt, 5)


        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        #if self.rs == True: --- Does not add value
        #    boxes, feats = transform_feats(boxes, feats, img_info["objects_id"])
        #    boxes = np.array(boxes, dtype=np.float32)
        #    feats = np.array(feats)


        if args.num_pos == 6:
            # Add width & height
            width = (boxes[:, 2] - boxes[:, 0]).reshape(-1,1)
            height = (boxes[:, 3] - boxes[:, 1]).reshape(-1,1)

            boxes = np.concatenate((boxes, width, height), axis=-1)

            # In UNITER they use 7 Pos Feats (See _get_img_feat function in their repo)
            if args.model == "U":
                boxes = np.concatenate([boxes, boxes[:, 4:5]*boxes[:, 5:]], axis=-1)


        # Pad Boxes
        if args.pad:
            if feats.shape[0] > args.num_features:

                feats = feats[:args.num_features, :]
                boxes = boxes[:args.num_features, :]

                
                num_b = boxes.shape[0]

            else:
                # Save num_b for attention mask lateron
                num_b = boxes.shape[0]

                boxes_pad = np.zeros((args.num_features, boxes.shape[1]), dtype=boxes.dtype)
                boxes_pad[:boxes.shape[0], :boxes.shape[1]] = boxes

                feats_pad = np.zeros((args.num_features, feats.shape[1]), dtype=feats.dtype)
                feats_pad[:feats.shape[0], :feats.shape[1]] = feats

                boxes = boxes_pad
                feats = feats_pad
        else:
            num_b = -1

        # Provide label (target) - From mmf_data
        if 'label' in datum:
            if int(datum["label"]) == 1:
                label = [0, 1]
            else:
                label = [1, 0]
            target = torch.tensor(datum["label"], dtype=torch.float) 
            label = torch.tensor(label, dtype=torch.float)
            # Return target for 1 label, label for 2
            return img_id, feats, boxes, num_b, text, label, target
        else:
            return img_id, feats, boxes, num_b, text

import random

#https://maelfabien.github.io/machinelearning/NLP_8/#random-insertion-ri


import albumentations as A

def transform_feats(boxes, feats, obj_ids):

    transform = A.Compose([
        #A.Cutout(num_holes=1, max_h_size=1, max_w_size=2048, fill_value=0, p=0.5)
        A.Blur(blur_limit=3, p=0.5),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['vg_dict']))

    transformed = transform(image=feats, bboxes=boxes, vg_dict=obj_ids)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']

    return transformed_bboxes, transformed_image

def swap_feats_gt(boxes, feats, boxes_gt, feats_gt):
    
    random_idx_1 = random.randint(0, len(boxes)-1)
    random_idx_2 = random.randint(0, len(boxes_gt)-1)
    counter = 0
    
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(boxes_gt)-1)
        counter += 1
        
        if counter > 3:
            return boxes, feats
    
    boxes[random_idx_1] = boxes_gt[random_idx_2]
    feats[random_idx_1] = feats_gt[random_idx_2]
    return boxes, feats

def random_swap_feats_gt(boxes, feats, boxes_gt, feats_gt, n):
    

    for _ in range(n):
        boxes, feats = swap_feats_gt(boxes, feats, boxes_gt, feats_gt)
    
    return boxes, feats

def swap_feats(boxes, feats):
    
    random_idx_1 = random.randint(0, len(boxes)-1)
    random_idx_2 = random_idx_1
    counter = 0
    
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(boxes)-1)
        counter += 1
        
        if counter > 3:
            return boxes, feats
    
    boxes[random_idx_1], boxes[random_idx_2] = boxes[random_idx_2], boxes[random_idx_1] 
    feats[random_idx_1], feats[random_idx_2] = feats[random_idx_2], feats[random_idx_1] 
    return boxes, feats

def random_swap_feats(boxes, feats, n):
    

    for _ in range(n):
        boxes, feats = swap_feats(boxes, feats)
    
    return boxes, feats

def swap_word(new_words):
    
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        
        if counter > 3:
            return new_words
    
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

def random_swap(words, n):
    
    words = words.split()
    new_words = words.copy()
    
    for _ in range(n):
        new_words = swap_word(new_words)
        
    sentence = ' '.join(new_words)
    
    return sentence

from nltk.corpus import wordnet 

def get_synonyms(word):
    """
    Get synonyms of a word
    """
    synonyms = set()
    
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)

def random_insertion(words, n):
    
    words = words.split()
    new_words = words.copy()
    
    for _ in range(n):
        add_word(new_words)
        
    sentence = ' '.join(new_words)
    return sentence

def add_word(new_words):
    
    synonyms = []
    counter = 0
    
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
        
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)


class MMFEvaluator:
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

    def dump_result(self, quesid2ans: dict, path):

        with open(path, "w") as f:
            result = []
            for img_id, ans in quesid2ans.items():
                result.append({"img_id": ques_id, "pred": ans})
            json.dump(result, f, indent=4, sort_keys=True)

    def dump_csv(self, quesid2ans: dict, quesid2prob: dict, path):

        d = {"id": [int(tensor) for tensor in quesid2ans.keys()], "proba": list(quesid2prob.values()), 
            "label": list(quesid2ans.values())}
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

            for key in ['img_id', 'img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
           
            # Uncomment & comment the below if boxes of variable len
            #boxes = item['num_boxes']

            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            try:
                for key, shape, dtype in decode_config:

                    item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                    try:
                        item[key] = item[key].reshape(shape)
                    # A box might be missing - Copying another box to the end
                    except:
                        item['num_boxes'] = boxes # Correct the number of boxes

                        print(key, shape, dtype)
                        print(item[key].shape)
                        if item[key].shape[0] == (shape[0] - 1):
                            arr = item[key][-1:].copy()
                        else:
                            # If it is a box, e.g. of shape 196 for 50 boxes
                            arr = item[key][-4:].copy()
                        item[key] = np.concatenate((item[key], arr))
                        item[key] = item[key].reshape(shape)

                    item[key].setflags(write=False)
            except:
                print(i)
                print(item)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data
