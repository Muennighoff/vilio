# Collection of pandas scripts that may be useful

import numpy as np
import pandas as pd
import os

from PIL import Image
import imagehash



def clean_data(data_path):
    pass




# Image hash functions: 
# https://content-blockchain.org/research/testing-different-image-hash-functions/

def hash_func(img_path):
    img = Image.open(img_path)
    img = np.asarray(img, dtype="int32") 
    hash_v = imagehash.phash(Image.open(img_path))
    return hash_v
    
def hash_func_crude(img_path):
    img = Image.open(img_path)
    img = np.asarray(img, dtype="int32")    
    # Going down the diagonal from ri up corner until no b/w/g value 
    row_val = 0
    col_val = -1
    try:
        while((not(255 > img[row_val, col_val, 0] > 1)) 
              or (not(img[row_val, col_val, 0] != img[row_val, col_val, 1] != img[row_val, col_val, 2]))):
            row_val += 1
            col_val -= 1    
    except:
        row_val = 0
        col_val = -1
        
        try:
            while(not(255 > img[row_val, col_val, 0] > 1)): 
                row_val += 1
                col_val -= 1  
        except:
            try:
                # It has no 3 channels
                while(not(255 > img[row_val, col_val] > 1)): 
                    row_val += 1
                    col_val -= 1 
                    
            except:
                print("3x Except: ", img_path)
                
            hash_v = str(img.shape[0]) + str(img.shape[1]) + str(img[row_val, col_val]) * 3
            return hash_v
            
    hash_v = str(img.shape[0]) + str(img.shape[1]) + str(img[row_val, col_val, 0]) + str(img[row_val, col_val, 1]) + str(img[row_val, col_val, 2])    
    return hash_v

def create_subdata(data_path="./data"):
    """
    data_path: Path to data folder containing all jsonl's & images under /img
    """

    traincleanex = pd.read_json(base_path + '/traincleanex.jsonl', lines=True)
    devseen = pd.read_json(base_path + '/devseen.jsonl', lines=True)
    test = pd.read_json(base_path + '/test.jsonl', lines=True)
    test_unseen = pd.read_json(base_path + '/test_unseen.jsonl', lines=True)

    df_dict = {'train': traincleanex, 'dev': devseen, 'test': test, 'test_unseen': test_unseen}
    full_dist = pd.concat([df.assign(identity=key) for key,df in df_dict.items()])

    # Create full path for easy image loading
    full_dist['full_path'] = full_dist['img'].apply(lambda x: data_path + str(x))

    full_dist['phash'] = full_dist['full_path'].apply(lambda x: hash_func(x))
    full_dist['crhash'] = full_dist['full_path'].apply(lambda x: hash_func_crude(x))

    full_dist["text_clean"] = full_dist["text"].str.replace("'", "")
    full_dist["text_clean"] = full_dist["text"].str.replace('"', '')

    full_dist["text_clean"] = full_dist["text_clean"].astype(str)
    full_dist["phash"] = full_dist["phash"].astype(str)
    full_dist["crhash"] = full_dist["crhash"].astype(str)


