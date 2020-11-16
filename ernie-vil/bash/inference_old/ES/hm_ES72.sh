#bash -x ./env.sh

# Allows for not having to copy the models to vilio/ernie-vil/data
loadfin=${1:-./data/LASTtrain.pth}
loadfin2=${2:-./data/LASTtraindev.pth}
loadfin3=${3:-./data/LASTtrain.pth}
loadfin4=${4:-./data/LASTtraindev.pth}
loadfin5=${5:-./data/LASTtrain.pth}
loadfin6=${6:-./data/LASTtraindev.pth}
loadfin7=${7:-./data/LASTtrain.pth}
loadfin8=${8:-./data/LASTtraindev.pth}

### ATT 72

mv ./data/hm/hm_vgattr10100.tsv ./data/hm/HM_gt_img.tsv
mv ./data/hm/hm_vgattr7272.tsv ./data/hm/HM_img.tsv


bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
$loadfin \
./data/log \
dev_seen ES72 False

# SUB 1

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
$loadfin2 \
./data/log \
dev_seens1 ES72 False

# SUB2

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
$loadfin3 \
./data/log \
dev_seens2 ES72 False

# SUB3

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
$loadfin4 \
./data/log \
dev_seens3 ES72 False

### TRAINDEV

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
$loadfin5 \
./data/log \
test_seen ES72 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
$loadfin5 \
./data/log \
test_unseen ES72 False

# SUB 1

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
$loadfin6 \
./data/log \
test_seens1 ES72 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
$loadfin6  \
./data/log \
test_unseens1 ES72 False

# SUB2

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
$loadfin7 \
./data/log \
test_seens2 ES72 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
$loadfin7 \
./data/log \
test_unseens2 ES72 False

# SUB3

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
$loadfin8 \
./data/log \
test_seens3 ES72 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/erniesmall/vocab.txt \
./data/erniesmall/ernie_vil_config.base.json \
$loadfin8 \
./data/log \
test_unseens3 ES72 True