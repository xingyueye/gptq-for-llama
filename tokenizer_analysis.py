from langdetect import detect
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/common/LM_data/model/bloom-new/bloom-7b1")
a = set()
for key in tokenizer.vocab:
    try:
        lang = detect(key)
        if lang in a:
            a[lang] += 1
        else:
            a[lang] = 1
    except:
        a['others'] += 1