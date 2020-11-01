# Collection of pandas scripts that may be useful

import numpy as np
import pandas as pd
import os

from PIL import Image
import imagehash


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

    ## Load all files
    train = pd.read_json(os.path.join(data_path, "train.jsonl"), lines=True, orient="records")
    dev_seen = pd.read_json(os.path.join(data_path, "dev_seen.jsonl"), lines=True, orient="records")
    # We validate with dev_seen throughout all experiments, so we only take the new data from dev_unseen add it to train and then discard dev_unseen
    dev_unseen = pd.read_json(os.path.join(data_path,"dev_unseen.jsonl"), lines=True, orient="records")
    dev_unseen = dev_unseen[~dev_unseen['id'].isin(dev_seen.id.values)].copy()

    test_seen = pd.read_json(os.path.join(data_path, "test_seen.jsonl"), lines=True, orient="records")
    test_unseen = pd.read_json(os.path.join(data_path, "test_unseen.jsonl"), lines=True, orient="records")

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

    pretrain = pd.concat([train, dev_seen, test_seen, dev_unseen, test_unseen])

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
    traincleandev = pd.concat([train, dev_unseen, dev_seen])
    traincleandev.to_json(path_or_buf=os.path.join(data_path, "traindev.jsonl"), orient='records', lines=True)


def create_subdata(data_path="./data"):
    """
    Splits the data into three equal-sized pots to perform subtraining. 

    data_path: Path to data folder containing all jsonl's & images under /img
    """
    # Check if the statement was already run and the necessary data exists:
    if os.path.exists(os.path.join(data_path, "train_ic.jsonl")):
        return

    train = pd.read_json(os.path.join(data_path + '/train.jsonl'), lines=True) # Note: This is the updated train, incl. data from dev_unseen
    dev = pd.read_json(os.path.join(data_path + '/dev_seen.jsonl'), lines=True)
    test = pd.read_json(os.path.join(data_path + '/test_seen.jsonl'), lines=True)
    test_unseen = pd.read_json(os.path.join(data_path + '/test_unseen.jsonl'), lines=True)

    df_dict = {'train': train, 'dev': dev, 'test': test, 'test_unseen': test_unseen}
    full_dist = pd.concat([df.assign(identity=key) for key,df in df_dict.items()])

    # Create full path for easy image loading
    full_dist['full_path'] = full_dist['img'].apply(lambda x: os.path.join(data_path, str(x)))

    full_dist['phash'] = full_dist['full_path'].apply(lambda x: phash(x))
    full_dist['crhash'] = full_dist['full_path'].apply(lambda x: crude_hash(x))

    full_dist["text_clean"] = full_dist["text"].str.replace("'", "")
    full_dist["text_clean"] = full_dist["text"].str.replace('"', '')

    full_dist["text_clean"] = full_dist["text_clean"].astype(str)
    full_dist["phash"] = full_dist["phash"].astype(str)
    full_dist["crhash"] = full_dist["crhash"].astype(str)

    full_dist["text_dups"] = full_dist["text_clean"].apply(lambda x: full_dist.loc[full_dist['text_clean'] == x].id.values)
    full_dist["phash_dups"] = full_dist["phash"].apply(lambda x: full_dist.loc[full_dist['phash'] == x].id.values)
    full_dist["crhash_dups"] = full_dist["crhash"].apply(lambda x: full_dist.loc[full_dist['crhash'] == x].id.values)

    dists = {}
    # Create ic dist to focus on data with similar text
    dists["ic"] = full_dist[full_dist["text_dups"].map(len) > 1].copy()

    # Create tc dist to focus on data with similar images
    dists["tc"] = full_dist[(full_dist["phash_dups"].map(len) + full_dist["crhash_dups"].map(len)) > 2].copy()

    # Create oc dist to focus on all the rest; i.e. on the diverse part
    dists["oc"] = full_dist[~((full_dist['id'].isin(dists["ic"].id.values)) | (full_dist['id'].isin(dists["tc"].id.values)))]

    for i in ["ic", "tc", "oc"]:
    
        train = dists[i].loc[dists[i].identity == "train"][["id", "img", "label", "text"]]
        train.to_json(data_path + '/train_' + i + '.jsonl', lines=True, orient="records")

        dev = dists[i].loc[dists[i].identity == "dev"][["id", "img", "label", "text"]]
        dev.to_json(data_path + '/dev_seen_' + i + '.jsonl', lines=True, orient="records")

        traindev = pd.concat([train, dev])
        traindev.to_json(data_path + '/traindev_' + i + '.jsonl', lines=True, orient="records")

        test = dists[i].loc[dists[i].identity == "test"][["id", "img", "text"]]
        test.to_json(data_path + '/test_seen_' + i + '.jsonl', lines=True, orient="records")

        test_unseen = dists[i].loc[dists[i].identity == "test_unseen"][["id", "img", "text"]]
        test_unseen.to_json(data_path + '/test_unseen_' + i + '.jsonl', lines=True, orient="records")


def unused():
    train = full_dist.loc[full_dist.identity == "train"][["id", "img", "label", "text"]]
    train.to_json(data_path + '/train_ic.jsonl', lines=True, orient="records")

    dev = full_dist.loc[full_dist.identity == "dev"][["id", "img", "label", "text"]]
    dev.to_json(data_path + '/devseen_ic.jsonl', lines=True, orient="records")

    traindev = pd.concat([train, dev])
    traindev.to_json(data_path + '/traindev_ic.jsonl', lines=True, orient="records")

    test = full_dist.loc[full_dist.identity == "test"][["id", "img", "text"]]
    test.to_json(data_path + '/test_ic.jsonl', lines=True, orient="records")

    test_unseen = full_dist.loc[full_dist.identity == "test_unseen"][["id", "img", "text"]]
    test_unseen.to_json(data_path + '/test_unseen_ic.jsonl', lines=True, orient="records")

    # Create tc dist to focus on data with similar images
    tc_dist = full_dist[(full_dist["phash_dups"].map(len) + full_dist["crhash_dups"].map(len)) > 2].copy()

    train = tc_dist.loc[tc_dist.identity == "train"][["id", "img", "label", "text"]]
    train.to_json(data_path + '/train_tc.jsonl', lines=True, orient="records")

    dev = tc_dist.loc[tc_dist.identity == "dev"][["id", "img", "label", "text"]]
    dev.to_json(data_path + '/dev_tc.jsonl', lines=True, orient="records")

    traindev = pd.concat([train, dev])
    traindev.to_json(data_path + '/traindev_tc.jsonl', lines=True, orient="records")

    test = tc_dist.loc[tc_dist.identity == "test"][["id", "img", "text"]]
    test.to_json(data_path + '/test_tc.jsonl', lines=True, orient="records")

    test_unseen = tc_dist.loc[tc_dist.identity == "test_unseen"][["id", "img", "text"]]
    test_unseen.to_json(data_path + '/test_unseen_tc.jsonl', lines=True, orient="records")

    # Create oc dist to focus on all the rest; i.e. on the diverse part
    oc_dist = full_dist[~((full_dist['id'].isin(ic_dist.id.values)) | (full_dist['id'].isin(tc_dist.id.values)))]

    train = oc_dist.loc[oc_dist.identity == "train"][["id", "img", "label", "text"]]
    train.to_json(data_path + '/train_oc.jsonl', lines=True, orient="records")

    dev = oc_dist.loc[oc_dist.identity == "dev"][["id", "img", "label", "text"]]
    dev.to_json(data_path + '/dev_oc.jsonl', lines=True, orient="records")

    traindev = pd.concat([train, dev])
    traindev.to_json(data_path + '/traindev_oc.jsonl', lines=True, orient="records")

    test = oc_dist.loc[oc_dist.identity == "test"][["id", "img", "text"]]
    test.to_json(data_path + '/test_oc.jsonl', lines=True, orient="records")

    test_unseen = oc_dist.loc[oc_dist.identity == "test_unseen"][["id", "img", "text"]]
    test_unseen.to_json(data_path + '/test_unseen_oc.jsonl', lines=True, orient="records")




