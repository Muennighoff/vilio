HM Conclusion


HM - Interesting directions not pursued:
- Grid features instead of boxes
	- https://drive.google.com/file/d/1j9QE6xBq7Al_92ylmQEO4Ufq4f5n3Awa/view
- Side-branches similar to the Inception Network
	- https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202
- Sub-models which support a bigger model
	- E.g. a sub CNN which only detects whether there is a person with down-syndrome in the image and then feeds this to the master model
- Re-pretraining with superior Bert models
	- Using Roberta, Albert & replicate pre-training e.g. for uniter should give a boost as HM is heavily text-dependent
- Negative Sampling (not really helpful for HM, as ROC AUC metric, not e.g. accuracy)
	- https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160986
	  

HM - Directions tried & scraped:
- Data Augmentation
	- Images via swaps etc / NLP via swaps, insertions - Theory: Images are often made up of two images and may even follow an order with the text, hence flips may hinder learning; NLP swaps showed some promise, but not stable -- I assume there is already enough noise in meme text
- Roc-Star Loss
	- A custom loss to optimize roc auc; Did help sometimes but not reliably, I decided to stick with CrossEntropy (NLLLoss + Logsoftmax)
- AWS / GCloud / Aliyun
	- Did not need the extra compute in the end; Aliyun no signup without phone number, GCloud did not approve limits
- Hyperparameter Tuning
	- Ended up not tuning a lot,as a changes in architecture / data have a far bigger impact than minor HP changes (I just sticked with standards used in previous similar problems)
- Correcting words
	- Set up a dict to replace e.g. nigga, niqqa etc on all occassions with nigger or maga with make amer.. etc - Did not help - Theory: Too few such occassions; There are differences which the NN learns, when sb uses the qq vs the gg version (To find such, I tried to use sound / character distances)
- Adding subwords
	- Often esp. with hashtages words are concatted, but at the same time many words in BERT Vocab are only known as non-subwords, hence the idea was to reinclude all words as subwords (i.e. increasing BERT vocab by about 20K) -- Did not find a good way to do this, even asked on Huggingface forums
- Pretrain BERT on jigsaw hate speech, text only
	- Did show an improvement, but not over pretrained v+l models --- pretraining a new v+l model with a jigsaw pre-trained bert could be show further improvements
- Not relying on pre-trained models
	- I implemented lots of newer language transformers within some of the VL models, such as Roberta or Albert for LXMERT. They did perform well, especially when also task specific pre-trained, but still came about absolute 1-3% short of the pretrained model. I think pretraining them from scratch (i.e. on COCO etc) could give a solid performance boost
- Adding Generated Captions to the text via [SEP]
	- Worked in the beginning, but only because my models lacked the ability to do good with the features
- Adding predicted objects & attributes to the text via [SEP] similar to OSCAR
	- Oddly did not help at all not even for OSCAR
- Using features with different min_boxes & max_boxes with padding 
	- Did not improve anything; Probably the model learns to focus on the first few boxes only anyways as they are sorted by confidence
- Random PS
	- Using the probability for each pseudolab to decide its label
- Label Smoothing
- Ensemble of ensembles
- Pseudolabels
	- Tried also smoothed pseudo-labels, i.e. taking the proba only; Did not work for me at all & since they were also not allowed in the competition thats good haha!
- Reiniting final layers & pooler
	- Did help initially, but not when pretraining - Could be used for the models we are not pretraining
- Conditioned bagging based on context
	- Tried to build models which would train on data e.g. relating to the them of world war II, and then combine those; worsened results - probably because topics are difficult to separate and have quite some overlap so it misses crucial data
- Adding new words
	- Against Jacob's recomm. I tried & learned. (My intuition was that BERT will never have seen words such as handjob in the given context, but probably finetuning is solving such issues already even though embeddings do not change during finetuning - Worsened rcac by 3% on a Bert-only)
	- https://github.com/google-research/bert/issues/9
- Flagging profanity words with a tag
	- https://github.com/RobertJGabriel/Google-profanity-words

HM - Directions tried & added:
- SWA/Stochastic Weight Averaging: Great when you want to train models “blind”. There was not a lot of data accessible so I decided to train models on all the data possible without validation, but use SWA to obtain stable results at the end. I still use models trained with validation, however to get score estimates and weights for the ensemble.
- Removing Duplicates & misclassified examples:
	- Using the IDs provided on the forum helped, after the updated train was released, just removing duplicates and keeping the rest worked best
	- Removed about 50 pure duplicates & added about 140 data from the new dev
- Predict dev based on train only > Use those dev preds to determine ensemble weights for test predictions based on train+dev
- Conditioned Bagging: The problem in my understanding: Due to conflicting labels (i.e. confounders with the same text) the model is much less confident about such predictions than single hate/non-hate predictions - Even though the model might correctly know which of those two is more hateful (i.e. it orders the two confounders when looked at in isolation correctly) it misplaces them on the “global roc auc scale”, i.e. when combined with all other predictions (e.g. too much to the right; to the left, due to e.g. the theme having stronger signals in one direction). I solved this problem by training one main model on the full data and three “children” on subsets of the data (which included e.g. more conflicting pairs). This allowed them to focus on that data in isolation and then make for a better score when recombining the predictions of the main model & the children. The idea of submodels was inspired by the Inception Network & I think it could be interesting to implement those submodels looking at different parts of the data into the main model. 
- Model-specific changes:
	- D
		- Update Bert Functions to be on par with huggingface as of 09/2020
	- O
		- Update Bert Functions to be on par with huggingface as of 09/2020
		- Task-specific Pre-training
	- U
		- Update Bert Functions to be on par with huggingface as of 09/2020
	- V
		- Update Bert Functions to be on par with huggingface as of 09/2020
		- Task-specific Pre-training
		- Configure with two token types and assign different token types to text & images
		- Layer Averaging
		- Multi-sample Dropout
	- X
		- Update Bert Functions to be on par with huggingface as of 09/2020
		- Task-specific Pre-training
	- E
		- Use extracted feats with 10 - 100 boxes as ground truth boxes (The original models were Pre-trained on hand labelled GT Boxes + the extracted feats)
		- Lots of small changes & upgrades from the original repo
