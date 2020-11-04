#bash -x ./env.sh

### ATT 36, Normal

cp ./data/hm/hm_vgattr10100.tsv ./data/hm/HM_gt_img.tsv
cp ./data/hm/hm_vgattr3636.tsv ./data/hm/HM_img.tsv

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./data/erniesmall/params \
train \
2500

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./output_hm/step_2500 \
./data/log \
val N36 False

#####

cp -r ./output_hm/step_1250 ./data/

######

bash run_finetuning.sh hm conf/hm/model_conf_hm \
../input/erniesmall/vocab.txt \
../input/erniesmall/ernie_vil_config.base.json \
./data/step_1250 \
trainic \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
../input/erniesmall/vocab.txt \
../input/erniesmall/ernie_vil_config.base.json \
./output_hm/step_1250 \
./data/log \
valic N36 False

# Same as above


# Add --exp to not just get pred.csv & modify in finetune
# Add subtrain to shell
# Add combine to shell (> Only for inference)
# 