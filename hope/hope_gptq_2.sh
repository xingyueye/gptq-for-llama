source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/local/env/gcc_dolphinfs.sh
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/lee/anaconda3/bin/activate
conda activate /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/lee/anaconda3/envs/gptq
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/lee/LLM/gptq-for-llama

#python opt_ln.py ../../../common/LM_data/model/opt/66b/ c4 --wbits 4 --act-order --sym --save_hf_model ckpt_hf_model/opt66b-4bit-act_order-sym-update_norm --update_norm
#python glm_ln.py ../../../common/LM_data/model/glm-130b-liangtao/glm-130b/ c4 --wbits 4 --act-order --sym --save_hf_model ckpt_hf_model/glm130b-4bit-act_order-sym-update_norm --update_norm
#python glm_ln.py ../../../common/LM_data/model/glm-130b-liangtao/glm-130b/ c4 --wbits 4 --act-order --sym --load_hf_model ckpt_hf_model/glm130b-4bit-act_order-sym-update_norm --update_norm --data_path ../../../common/LM_data/data/NLP/lambada_openai/lambada_test.jsonl
#python opt_ln.py ../../../common/LM_data/model/opt/66b/ c4 --wbits 4 --act-order --sym --load_hf_model ckpt_hf_model/opt66b-4bit-act_order-sym-update_norm --update_norm --data_path ../../../common/LM_data/data/NLP/lambada_openai/lambada_test.jsonl
python bloom_ln_mean.py ../../../common/LM_data/model/bloom-new/bloom-176b/ c4 --wbits 4 --act-order --sym  --lr 1e-6 --save_hf_model ./ckpt_hf_model/bloom176b-4bit-act_order-sym-update_norm --update_norm
#python llama_ln.py ../../../common/LM_data/model/llama-65b-hf c4 --wbits 4 --act-order --sym  --save_hf_model ./ckpt_hf_model/llama65b-4bit-act_order-sym-update_norm
python bloom_ln_mean.py ../../../common/LM_data/model/bloom-new/bloom-176b/ c4 --wbits 4 --act-order --sym  --lr 1e-6 --load_hf_model ./ckpt_hf_model/bloom176b-4bit-act_order-sym-update_norm --update_norm --data_path ../../../common/LM_data/data/NLP/lambada_openai/lambada_test.jsonl
#python llama_ln.py ../../../common/LM_data/model/llama-65b-hf c4 --wbits 4 --act-order --sym  --load_hf_model ./ckpt_hf_model/llama65b-4bit-act_order-sym-update_norm  --data_path ../../../common/LM_data/data/NLP/lambada_openai/lambada_test.jsonl
#python llama_inference.py ../../../common/LM_data/model/llama-65b-hf/ --text 'The weight matrix within the model would be compressed, which means it may contain errors. It is crucial to provide appropriate calibration data to ensure optimal performance. The following is an example of good calibration sample: '  --max_length 2048
#python bloom_inference.py ../../../common/LM_data/model/bloom-new/bloom-176b/ --text 'The weight matrix within the model would be compressed, which means it may contain errors. It is crucial to provide appropriate calibration data to ensure optimal performance. The following is an example of good calibration sample: '  --max_length 2048 --save_npy 'bloom-176b-gen-calib-128x2048.npy'
