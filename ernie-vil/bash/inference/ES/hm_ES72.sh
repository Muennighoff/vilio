#bash -x ./env.sh

# Allows for not having to copy the models to vilio/ernie-vil/data
loadfin=${1:-./data/LASTtrain.pth}
loadfin2=${2:-./data/LASTtraindev.pth}

### ATT 72

mv ./data/hm/hm_vgattr10100.tsv ./data/hm/HM_gt_img.tsv
mv ./data/hm/hm_vgattr7272.tsv ./data/hm/HM_img.tsv


bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
$loadfin \
./data/log \
dev_seen ES72 False

### TRAINDEV

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
$loadfin2 \
./data/log \
test_seen ES72 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
$loadfin2 \
./data/log \
test_unseen ES72 False