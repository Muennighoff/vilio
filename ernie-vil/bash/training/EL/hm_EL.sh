#bash -x ./env.sh

bash ./bash/training/EL/hm_EL36.sh

bash ./bash/training/EL/hm_ELVCR36.sh

bash ./bash/training/EL/hm_ELV50.sh

bash ./bash/training/EL/hm_EL72.sh

bash ./bash/training/EL/hm_ELVCR72.sh

# Simple Average
python utils/ens.py --enspath ./data/hm/ --enstype sa --exp EL365072