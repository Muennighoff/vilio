#bash -x ./env.sh

bash ./bash/training/ES/hm_ES36.sh

bash ./bash/training/ES/hm_ESVCR36.sh

bash ./bash/training/ES/hm_ESV50.sh

bash ./bash/training/ES/hm_ES72.sh

bash ./bash/training/ES/hm_ESVCR72.sh

# Simple Average
python utils/ens.py --enspath ./data/hm/ --enstype sa --exp ES365072