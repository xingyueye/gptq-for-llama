source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/local/env/gcc_dolphinfs.sh
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/lee/anaconda3/bin/activate
conda activate /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/lee/anaconda3/envs/gptq
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/lee/LLM/lm-evaluation-harness

#python main.py --model hf-causal-experimental --model_args pretrained=../gptq-for-llama/ckpt_hf_model/bloom/bloom176b-2bit-gs64-act_order-sym-update_norm/,dtype=float16,batch_size=1,max_length=2048 --tasks hellaswag,piqa,winogrande,openbookqa,rte,mrpc,qnli,boolq,cb,copa,wic --output_path outputs/bloom176-2bit.txt
python main.py --model hf-causal-experimental --model_args pretrained=../gptq-for-llama/ckpt_hf_model/bloom/bloom176b-2bit-gs64-act_order-sym-update_norm_3/,dtype=float16,batch_size=1,max_length=2048 --tasks hellaswag,piqa,winogrande,openbookqa,rte,mrpc,qnli,boolq,cb,copa,wic --output_path outputs/bloom176-2bit-mean_3.txt
