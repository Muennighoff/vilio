#bash -x ./env.sh

### ATT 36

mv ./data/hm/hm_vgattr10100.tsv ./data/hm/HM_gt_img.tsv
mv ./data/hm/hm_vgattr3636.tsv ./data/hm/HM_img.tsv

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/erniesmallvcr/vocab.txt \
./data/erniesmallvcr/ernie_vil_config.base.json \
./data/erniesmallvcr/params \
train \
2500

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmallvcr/vocab.txt \
./data/erniesmallvcr/ernie_vil_config.base.json \
./output_hm/step_2500train \
./data/log \
dev_seen ESVCR36 False

### TRAINDEV

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/erniesmallvcr/vocab.txt \
./data/erniesmallvcr/ernie_vil_config.base.json \
./data/erniesmallvcr/params \
traindev \
2500

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmallvcr/vocab.txt \
./data/erniesmallvcr/ernie_vil_config.base.json \
./output_hm/step_2500traindev \
./data/log \
test_seen ESVCR36 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmallvcr/vocab.txt \
./data/erniesmallvcr/ernie_vil_config.base.json \
./output_hm/step_2500traindev \
./data/log \
test_unseen ESVCR36 False