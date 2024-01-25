source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/local/env/gcc_dolphinfs.sh
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/lee/anaconda3/bin/activate
conda activate /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/lee/anaconda3/envs/gptq
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/lee/LLM/lm-evaluation-harness

python main.py --model hf-causal-experimental --model_args pretrained=../gptq-for-llama/ckpt_hf_model/opt66b/opt66b-4bit-act_order-sym-update_norm/,dtype=float16,batch_size=1,max_length=2048 --tasks hellaswag,piqa,winogrande,openbookqa,rte,mrpc,qnli,boolq,cb,copa,wic --output_path outputs/opt66b-4bit.txt
