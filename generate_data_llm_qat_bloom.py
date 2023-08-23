from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import sys
import os
import numpy as np
import random

rand_seed = sys.argv[1]
file_name = sys.argv[2]
random.seed(rand_seed)



from langdetect import detect

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/common/LM_data/model/bloom-new/bloom-7b1")
print("Tokenizer loaded!")
print("Loading model")
model = AutoModelForCausalLM.from_pretrained("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/common/LM_data/model/bloom-new/bloom-7b1")
print("Model loaded!")

n_vocab = 500 # number of initial tokens for synthesizing data on each GPU.

# i_start = sys.argv[1]
i_start = 8
if os.path.exists("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl"):
    with open("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl", "r") as f:
        lines = f.readlines()
        inner_loop = len(lines) % n_vocab
        outer_loop = len(lines) // n_vocab
else:
    inner_loop = 0
    outer_loop = 0

model.cuda()
gen_inp = []
seq = 128
max_length = 2048

key_list = list(tokenizer.vocab.keys())

# validList = ['ar', 'af', 'en', 'es', 'pt', 'zh-cn', 'fr', 'vi']  # 7 language
validList = ['en', 'es', 'pt', 'zh-cn', 'fr']  # 5 language
def isNotValid(char):
    try:
        lang = detect(char)
        return lang not in validList
    except:
        return True

for i in range(seq):
    print(i)
    # rand_key = random.choice(key_list)
    # while isNotValid(rand_key):
    #     rand_key = random.choice(key_list)
    # input_ids = torch.tensor([[tokenizer.vocab[rand_key]]]).cuda()
    idx = random.randint(0, 250680)
    input_ids = torch.tensor([[idx]]).cuda()
    outputs1 = model.generate(input_ids, do_sample=False, max_length=5)
    outputs = model.generate(outputs1, do_sample=True, min_length=max_length, max_length=max_length)
    gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # text_dict = {"text" : gen_text[0]}
    with open("gen_data/" + file_name + ".jsonl", "a") as f:
        f.write(json.dumps(gen_text))
        f.write('\n')
    if outputs.shape[-1] == max_length:
        gen_inp.append(outputs.cpu().numpy())

gen_inp = np.stack(gen_inp, axis=0)
np.save("gen_data/" + file_name + ".npy", gen_inp)