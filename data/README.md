


Files:

traincleanex - Train, new labels, dups removed, extra data from dev unseen
traincleanex2 - Same as above but data from dev seen to validate on dev unseen

train / test _s > Data for siamesei.e. with dups

psxx > Pseudo exps

Features / LMDB file should be placed here.

...A > Without Apostrophes











OLD: (remove after comp)


Place the data in this folder without any subfolders. 
For custom datafolders adjust the paths in mmf.py.
For custom lmdb extraction, adjust the paths in lmdb_conversion.py. 

For hateful memes, we expect the following in this folder:
- train.jsonl
- dev.jsonl
- test.jsonl
- detectron.lmdb 
- vocabulary.txt

Then run lmdb_conversion.py to convert the lmdb to npy features (it will create a features folder in data).

Command: python lmdb_conversion.py 

(For modifying, simply change the default which is set to: --mode extract --lmdb_path data/detectron.lmdb  --features_folder data/features_ex)


The vocabulary file should have the same size as our models embedding size & the necessary tokens should be added. The simplest way is to download the respective vocab file:
- https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt

Some models require more than just a txt file, e.g.:
- https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-vocab.json
- https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-merges.txt




Then just replace UNUSED tokens with the specified words for optimal performance. We use substrings (##WORD), normal words (WORD) and auxiliary tokens, disguised as normal words, (auxNAME). 