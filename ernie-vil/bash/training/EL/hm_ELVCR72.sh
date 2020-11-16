#bash -x ./env.sh

### ATT 72, Normal

mv ./data/hm/hm_vgattr10100.tsv ./data/hm/HM_gt_img.tsv
mv ./data/hm/hm_vgattr7272.tsv ./data/hm/HM_img.tsv

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielargevcr/vocab.txt \
./data/ernielargevcr/ernie_vil.large.json \
./data/ernielargevcr/params \
train \
2500

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielargevcr/vocab.txt \
./data/ernielargevcr/ernie_vil.large.json \
./output_hm/step_2500train \
./data/log \
dev_seen ELVCR72 False


##################### TRAINDEV


bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielargevcr/vocab.txt \
./data/ernielargevcr/ernie_vil.large.json \
./data/ernielargevcr/params \
traindev \
2500

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielargevcr/vocab.txt \
./data/ernielargevcr/ernie_vil.large.json \
./output_hm/step_2500traindev \
./data/log \
test_seen ELVCR72 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielargevcr/vocab.txt \
./data/ernielargevcr/ernie_vil.large.json \
./output_hm/step_2500traindev \
./data/log \
test_unseen ELVCR72 False

