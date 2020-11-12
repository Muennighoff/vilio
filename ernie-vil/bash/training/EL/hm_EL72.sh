#bash -x ./env.sh

### ATT 72

mv ./data/hm/hm_vgattr10100.tsv ./data/hm/HM_gt_img.tsv
mv ./data/hm/hm_vgattr7272.tsv ./data/hm/HM_img.tsv

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./data/ernielarge/params \
train \
2500

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_2500train \
./data/log \
dev_seen EL72 False

# Save Space

rm -r ./data/hm/img

# SUB 1

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250train \
trains1 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250trains1 \
./data/log \
dev_seens1 EL72 False

# SUB2

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250train \
trains2 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250trains2 \
./data/log \
dev_seens2 EL72 False

# SUB3

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250train \
trains3 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250trains3 \
./data/log \
dev_seens3 EL72 False

##################### TRAINDEV


bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./data/ernielarge/params \
traindev \
2500

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_2500traindev \
./data/log \
test_seen EL72 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_2500traindev \
./data/log \
test_unseen EL72 False

# Midsave

#cp -r ./output_hm/step_1250 ./data/

# SUB 1

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250traindev \
traindevs1 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250traindevs1 \
./data/log \
test_seens1 EL72 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250traindevs1 \
./data/log \
test_unseens1 EL72 False

# SUB2

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250traindev \
traindevs2 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250traindevs2 \
./data/log \
test_seens2 EL72 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250traindevs2 \
./data/log \
test_unseens2 EL72 False

# SUB3

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250traindev \
traindevs3 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250traindevs3 \
./data/log \
test_seens3 EL72 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250traindevs3 \
./data/log \
test_unseens3 EL72 True