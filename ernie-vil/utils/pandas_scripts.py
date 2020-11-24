# Collection of pandas scripts that may be useful
import numpy as np
import pandas as pd
import os

from PIL import Image
import imagehash

from sklearn.metrics import roc_auc_score


# Image hash functions: 
# https://content-blockchain.org/research/testing-different-image-hash-functions/

def phash(img_path):
    phash = imagehash.phash(Image.open(img_path))
    return phash

def crude_hash(img_path):
    """
    The function generates a hash based on simple comparisons such as dimensions of an image
    """
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

### Data Cleaning

# The HM Dataset is very noisy:
#  In the first version of the dataset there were many duplicates with conflicting labels
#  In the second version, the conflicting labels have all been resolved, yet the duplicates remain
def clean_data(data_path="./data"):
    """
    Cleans the HM train & dev data.
    Outputs traindev & pretrain data.

    data_path: Path to folder with train.jsonl, dev_unseen.jsonl, dev_seen.jsonl
    """
    # Check if the statement was already run and the necessary data exists:
    if os.path.exists(os.path.join(data_path, "pretrain.jsonl")):
        return
    else:
        print("Preparing...")

    ## Load all files
    train = pd.read_json(os.path.join(data_path, "train.jsonl"), lines=True, orient="records")
    dev_seen = pd.read_json(os.path.join(data_path, "dev_seen.jsonl"), lines=True, orient="records")
    # We validate with dev_seen throughout all experiments, so we only take the new data from dev_unseen add it to train and then discard dev_unseen
    dev_unseen = pd.read_json(os.path.join(data_path,"dev_unseen.jsonl"), lines=True, orient="records")
    dev_unseen = dev_unseen[~dev_unseen['id'].isin(dev_seen.id.values)].copy()

    ## Clean training data
    df_dict = {'train': train, 'dev_seen': dev_seen, 'dev_unseen': dev_unseen}
    train_dist = pd.concat([df.assign(identity=key) for key,df in df_dict.items()])

    # Hash images
    train_dist['full_path'] = train_dist['img'].apply(lambda x: os.path.join(data_path, str(x)))
    train_dist['hash'] = train_dist['full_path'].apply(lambda x: phash(x))
    train_dist['hash_cr'] = train_dist['full_path'].apply(lambda x: crude_hash(x))

    # Find dups among images & text
    train_dist['img_dup0'] = train_dist.duplicated(subset='hash', keep=False)
    train_dist['img_dup1'] = train_dist.duplicated(subset='hash_cr', keep=False)
    train_dist["txtdup"] = train_dist.duplicated(subset='text', keep=False)

    # Identify 100% dups
    hash_df = train_dist.hash.value_counts().reset_index(name="counter")
    hash_df = hash_df.loc[hash_df['counter'] > 1]
    hash_df["hash"] = hash_df["index"].astype(str)

    rmv_ids = []

    for h in hash_df['index'].astype(str).values:
        hash_group = train_dist.loc[train_dist.hash.astype(str) == h]
        
        txtdup = hash_group.duplicated(subset='text', keep=False).values
        imgdup1 = hash_group.duplicated(subset='hash_cr', keep=False).values
        
        if (True in txtdup) and (True in imgdup1):
            if len(txtdup) == 2:
                if hash_group.label.values[0] == hash_group.label.values[1]:
                    rmv_ids.append(hash_group.id.values[0]) # They are 100% identical, we'll just rmv the first
                else:
                    print("Labs not the same:", hash_group.id.values, hash_group.label.values) # None here
            else:
                # About 15 examples which are in the below ISIS & ADD lists.
                pass

    ISIS_dups = [35097, 97542, 91562, 71368, 29013, 85173, 15072, 1348, 70269, 36804, 68954, 91270, 64781, 96078, 97162, 34518, 17834,
                 31408, 56134, 68231, 98517, 27156, 10793, 82169, 25780, 25913, 95401, 94850, 50624, 92845, 58732]
    ADD_dups = [54981, 71903, 69087]

    rmv_ids.extend(ISIS_dups)
    rmv_ids.extend(ADD_dups)
    
    ## Output all files we need
    
    # a) Pretrain file for ITM & LM pre-training
    pretrain = pd.concat([train, dev_seen, dev_unseen])

    # The following ids throw some dimension error when pre-training; we can afford to skip them
    dim_error = [63805, 73026, 16845, 27058]
    pretrain = pretrain[~pretrain['id'].isin(dim_error)]
    # Note: The following is slightly different than I did in the original submission, but it performs in fact better
    pretrain["label"].fillna(0, inplace=True)
    pretrain.to_json(path_or_buf=os.path.join(data_path, "pretrain.jsonl"), orient='records', lines=True)

    # b) Cleaned Train + unused data from dev_unseen (All duplicates are in train, hence the following suffices)
    train = train[~train['id'].isin(rmv_ids)].copy()
    trainclean = pd.concat([train, dev_unseen])
    trainclean.to_json(path_or_buf=os.path.join(data_path, "train.jsonl"), orient='records', lines=True)

    # c) Cleaned Train + unused data from dev_unseen + dev_seen
    traincleandev = pd.concat([dev_seen, train, dev_unseen, dev_seen])
    traincleandev.to_json(path_or_buf=os.path.join(data_path, "traindev.jsonl"), orient='records', lines=True)


def double_data(data_path="./data"):
    """
    Takes the data and pastes it on to the end. This ensures the data is formatted correctly for E-Models.
    It also creates a dummy column "label" which is just set to 0. 
    The test data is only used at inference.
    """
    data = ["train", "traindev", "dev_seen", "test_seen", "test_unseen"]

    preds = {}
    for csv in sorted(os.listdir(data_path)):
        if any(d in csv for d in data) and ("jsonl" in csv) and ("long" not in csv):
            df = pd.read_json(os.path.join(data_path, csv), lines=True, orient="records")
            if "test" in csv:
                df["label"] = 0
                df.loc[0, "label"] = 1
            pd.concat([df, df[:int(0.3*len(df))]]).to_json(os.path.join(data_path, csv[:-6] + "long" + ".jsonl"), lines=True, orient="records")