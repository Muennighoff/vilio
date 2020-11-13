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

# Save Space
rm -r ./data/hm/img

# SUB 1

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielargevcr/vocab.txt \
./data/ernielargevcr/ernie_vil.large.json \
./output_hm/step_1250train \
trains1 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielargevcr/vocab.txt \
./data/ernielargevcr/ernie_vil.large.json \
./output_hm/step_1250trains1 \
./data/log \
dev_seens1 ELVCR72 False

# SUB2

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielargevcr/vocab.txt \
./data/ernielargevcr/ernie_vil.large.json \
./output_hm/step_1250train \
trains2 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielargevcr/vocab.txt \
./data/ernielargevcr/ernie_vil.large.json \
./output_hm/step_1250trains2 \
./data/log \
dev_seens2 ELVCR72 False

# SUB3

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielargevcr/vocab.txt \
./data/ernielargevcr/ernie_vil.large.json \
./output_hm/step_1250train \
trains3 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielargevcr/vocab.txt \
./data/ernielargevcr/ernie_vil.large.json \
./output_hm/step_1250trains3 \
./data/log \
dev_seens3 ELVCR72 False