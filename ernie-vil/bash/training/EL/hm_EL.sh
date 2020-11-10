#bash -x ./env.sh

bash ./bash/EL/hm_EL36.sh

bash ./bash/EL/hm_ELV50.sh

bash ./bash/EL/hm_EL72.sh

bash ./bash/EL/hm_ELVCR36.sh

bash ./bash/EL/hm_ELVCR72.sh

# Simple Average
python utils/ens.py --enspath ./data/hm/ --enstype sa --exp ES365072