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
./output_hm/step_2500train \
./data/log \
dev_seen ES36 False

# Midsave

#cp -r ./output_hm/step_1250 ./data/

# SUB 1

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./output_hm/step_1250train \
trains1 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./output_hm/step_1250trains1 \
./data/log \
dev_seens1 ES36 False

# SUB2

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./data/step_1250 \
trains2 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./output_hm/step_1250 \
./data/log \
dev_seens2 ES36 False

# SUB3

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./data/step_1250 \
trains3 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./output_hm/step_1250 \
./data/log \
dev_seens3 ES36 False

##################### TRAINDEV


bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./data/erniesmall/params \
traindev \
2500

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./output_hm/step_2500 \
./data/log \
test_seen ES36 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./output_hm/step_2500 \
./data/log \
test_unseen ES36 False

# Midsave

cp -r ./output_hm/step_1250 ./data/

# SUB 1

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./data/step_1250 \
traindevs1 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./output_hm/step_1250 \
./data/log \
test_seens1 ES36 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./output_hm/step_1250 \
./data/log \
test_unseens1 ES36 False

# SUB2

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./data/step_1250 \
traindevs2 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./output_hm/step_1250 \
./data/log \
test_seens2 ES36 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./output_hm/step_1250 \
./data/log \
test_unseens2 ES36 False

# SUB3

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./data/step_1250 \
traindevs3 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./output_hm/step_1250 \
./data/log \
test_seens3 ES36 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
./output_hm/step_1250 \
./data/log \
test_unseens3 ES36 True