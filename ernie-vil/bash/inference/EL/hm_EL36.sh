#bash -x ./env.sh

# Allows for not having to copy the models to vilio/ernie-vil/data
loadfin=${1:-./data/LASTtrain.pth}
loadfin2=${2:-./data/LASTtraindev.pth}

### ATT 36

mv ./data/hm/hm_vgattr10100.tsv ./data/hm/HM_gt_img.tsv
mv ./data/hm/hm_vgattr3636.tsv ./data/hm/HM_img.tsv


bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
$loadfin \
./data/log \
dev_seen EL36 False

##################### TRAINDEV

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
$loadfin2 \
./data/log \
test_seen EL36 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
$loadfin2 \
./data/log \
test_unseen EL36 False