import numpy as np
import copy
import pickle
import base64

import json
import os

## NOTE: We can use 10100 TSV here, as feats are later padded

class ImageFeaturesH5Reader(object):
    def __init__(self, features_path, jsonl_path="./data/hm/pretrain.jsonl"):
        """
        features_path: Path with tsv & jsonl's
        """
        #self.splits = splits.split(",")

        # Loading datasets to data
        #self.raw_data = []

        #for split in self.splits:
        #    path = os.path.join("data/", f"{split}.jsonl")
        #    self.raw_data.extend(
        #            [json.loads(jline) for jline in open(path, "r").read().split('\n')]
        #    )

        entries = []
        entries.extend(
            [json.loads(jline) for jline in open(jsonl_path, "r").read().split('\n')]
            )

        id2datum = {datum["id"]: datum for datum in entries}

        #print("Load %d data from split(s) %s." % (len(self.raw_data)))

        # Loading datasets to data

        # List to dict (for evaluation and others)
        #self.id2datum = {datum["id"]: datum for datum in self.raw_data}

        # Loading detection features to img_data
        img_data = []
        img_data.extend(load_obj_tsv(features_path, id2datum.keys()))


        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            # Adding int here to convert 0625 to 625
            self.imgid2img[int(img_datum['img_id'])] = img_datum

        # Only keep the data with loaded image features
        #self.data = []
        #for datum in self.raw_data:
            # In HM the Img Id field is simply "id"
        #    if datum['id'] in self.imgid2img:
        #        self.data.append(datum)

        #print("Use %d data in torch dataset" % (len(self.data)))
        #print()

    def __len__(self):
        return len(self.imgid2img)

    def __getitem__(self, image_id: int):
        """
        We give it the img_id here, not just any item index
        """

        # Get image info
        item = self.imgid2img[int(image_id)]
        num_boxes = item['num_boxes']
        image_h, image_w = item['img_h'], item['img_w']

        # Decoding already done in TSV loading
        #features = np.frombuffer(base64.b64decode(item["features"]), dtype=np.float32).reshape(num_boxes, 2048)
        #boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(num_boxes, 4)
        features = item["features"]
        boxes = item['boxes']

        g_feat = np.sum(features, axis=0) / num_boxes
        num_boxes = num_boxes + 1
        features = np.concatenate([np.expand_dims(g_feat, axis=0), features], axis=0)
        image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
        image_location[:, :4] = boxes
        image_location[:, 4] = (image_location[:, 3] - image_location[:, 1]) *   \
                (image_location[:, 2] - image_location[:, 0]) / (float(image_w) * float(image_h))

        image_location_ori = copy.deepcopy(image_location)
        image_location[:, 0] = image_location[:, 0] / float(image_w)
        image_location[:, 1] = image_location[:, 1] / float(image_h)
        image_location[:, 2] = image_location[:, 2] / float(image_w)
        image_location[:, 3] = image_location[:, 3] / float(image_h)

        g_location = np.array([0, 0, 1, 1, 1])
        image_location = np.concatenate([np.expand_dims(g_location, axis=0), image_location], axis=0)

        g_location_ori = np.array([0, 0, image_w, image_h, image_w * image_h])
        image_location_ori = np.concatenate([np.expand_dims(g_location_ori, axis=0), image_location_ori], axis=0)

        data_json = {"features": features,
                     "num_boxes": num_boxes,
                     "image_location": image_location,
                     "image_location_ori": image_location_ori
            }
        return data_json


# TSV EXTRACTION

import sys
import csv
import base64
import time


csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

def load_obj_tsv(fname, ids, topk=None):
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
                    # In 1 out of 10K cases, one box appears to be missing -- we copy the last & concat it back to the end
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