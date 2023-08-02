import os
import numpy as np
import torch
from datasets import load_dataset, load_from_disk

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model):
    # from datasets import load_dataset
    # traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    # testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    traindata = load_from_disk('datasets/wikitext/wikitext-2-raw-v1/train')
    testdata = load_from_disk('datasets/wikitext/wikitext-2-raw-v1/validation')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model):
    # # from datasets import load_dataset
    # traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    # valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    traindata = load_from_disk('datasets/ptb_text_only/penn_treebank/train')
    valdata = load_from_disk('datasets/ptb_text_only/penn_treebank/validation')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model):
    # from datasets import load_dataset, load_from_disk
    # traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', use_auth_token=False)
    # valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', use_auth_token=False)
    traindata = load_from_disk('datasets/allenai/c4/train')
    valdata = load_from_disk('datasets/allenai/c4/validation')

    from transformers import AutoTokenizer
    try:
        if 'glm' in model:
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        if "glm" in model:
            if hasattr(tokenizer, 'gmask_token_id'):
                inp[:, -2] = tokenizer.gmask_token_id
            else:
                inp[:, -2] = 150001
            inp[:, -1] = tokenizer.bos_token_id
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_prompt_tokens(nsamples, seed, seqlen, model, real_model):

    from transformers import AutoTokenizer
    try:
        if 'glm' in model:
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    prompt_text = "Please carefully examine the weight matrix within the model, as it may contain errors. It is crucial to verify its accuracy and make any necessary adjustments to ensure optimal performance."
    data_path = model + 'Gen_prompt_data_v1.npy'
    trainloader = []
    if os.path.isfile(data_path):
        data = np.load(data_path)
        for i in range(nsamples):
            inp = torch.tensor(data[i])
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    else:

        import random
        import string
        random.seed(seed)
        trainloader = []

        key_list = list(tokenizer.vocab.keys())
        value_list = list(tokenizer.vocab.values())

        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(next(real_model.parameters()).device)

        gen_inp = []
        for _ in range(nsamples):
            # while True:
            #     idx = random.randint(0, tokenizer.vocab_size)
            #     s = key_list[idx]
            #     if isEnglish(s):
            #         idx = tokenizer.vocab[s]
            #         break
            # rand_inp = [idx]
            # print("[GENERATE DATA]: {}".format(s))
            # # input_ids = tokenizer.encode(rand_inp, return_tensors="pt").to(next(real_model.parameters()).device)
            # input_ids = torch.tensor(rand_inp).reshape(1, -1).to(next(real_model.parameters()).device)
            with torch.no_grad():
                inp = real_model.generate(
                    input_ids,
                    do_sample=True,
                    min_length=seqlen,
                    max_length=seqlen,
                    top_p=0.95,
                    temperature=0.8,
                )
            if "glm" in model:
                if hasattr(tokenizer, 'gmask_token_id'):
                    inp[:, -2] = tokenizer.gmask_token_id
                else:
                    inp[:, -2] = 150001
                inp[:, -1] = tokenizer.bos_token_id
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

            gen_inp.append(inp.cpu().numpy())
        gen_inp = np.stack(gen_inp, axis=0)

        np.save(data_path, gen_inp)

    valenc = None
    return trainloader, valenc

def get_random_generalize_tokens(nsamples, seed, seqlen, model, real_model, gen_data=None):

    from transformers import AutoTokenizer
    try:
        if 'glm' in model:
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    data_path = model + 'Gen_data_seed_{}.npy'.format(seed) if gen_data is None else gen_data
    trainloader = []
    if os.path.isfile(data_path):
        data = np.load(data_path)
        for i in range(nsamples):
            inp = torch.tensor(data[i])
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        print("Successfully Load Generated Calibration Data: {}".format(data_path))
    else:
        import random
        import string
        random.seed(seed)
        trainloader = []

        key_list = list(tokenizer.vocab.keys())
        value_list = list(tokenizer.vocab.values())

        def isEnglish(s):
            for c in s:
                if not c.isalpha() or c not in string.ascii_letters:
                    return False
            return True

        gen_inp = []
        for _ in range(nsamples):
            while True:
                idx = random.randint(0, tokenizer.vocab_size)
                s = key_list[idx]
                if isEnglish(s):
                    idx = tokenizer.vocab[s]
                    break
            rand_inp = [idx]
            print("[GENERATE DATA]: {}".format(s))
            # input_ids = tokenizer.encode(rand_inp, return_tensors="pt").to(next(real_model.parameters()).device)
            input_ids = torch.tensor(rand_inp).reshape(1, -1).to(next(real_model.parameters()).device)
            with torch.no_grad():
                inp = real_model.generate(
                    input_ids,
                    do_sample=True,
                    min_length=seqlen,
                    max_length=seqlen,
                    top_p=0.95,
                    temperature=0.8,
                )
            if "glm" in model:
                if hasattr(tokenizer, 'gmask_token_id'):
                    inp[:, -2] = tokenizer.gmask_token_id
                else:
                    inp[:, -2] = 150001
                inp[:, -1] = tokenizer.bos_token_id
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

            gen_inp.append(inp.cpu().numpy())
        gen_inp = np.stack(gen_inp, axis=0)
        np.save(data_path, gen_inp)

    valenc = None
    return trainloader, valenc

def get_random_generalize_tokens_2stages(nsamples, seed, seqlen, model, real_model, gen_data=None):

    from transformers import AutoTokenizer
    try:
        if 'glm' in model:
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    data_path = model + 'Gen_data_seed_{}.npy'.format(seed) if gen_data is None else gen_data
    # data_path = 'llama-65b-gen-calib-128x2048.npy'
    # data_path = 'bloom-7b-gen-calib-128x2048.npy'
    # data_path = gen_data
    trainloader = []
    if os.path.isfile(data_path):
        data = np.load(data_path)
        for i in range(nsamples):
            inp = torch.tensor(data[i])
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        print("Successfully Load Generated Calibration Data: {}".format(data_path))
    else:
        import random
        import string
        random.seed(seed)
        trainloader = []

        key_list = list(tokenizer.vocab.keys())
        value_list = list(tokenizer.vocab.values())

        def isEnglish(s):
            for c in s:
                if not c.isalpha() or c not in string.ascii_letters:
                    return False
            return True

        gen_inp = []
        for _ in range(nsamples):
            while True:
                idx = random.randint(0, tokenizer.vocab_size)
                s = key_list[idx]
                if isEnglish(s):
                    idx = tokenizer.vocab[s]
                    break
            rand_inp = [idx]
            print("[GENERATE DATA]: {}".format(s))
            # input_ids = tokenizer.encode(rand_inp, return_tensors="pt").to(next(real_model.parameters()).device)
            input_ids = torch.tensor(rand_inp).reshape(1, -1).to(next(real_model.parameters()).device)
            with torch.no_grad():
                inp = real_model.generate(
                    input_ids,
                    do_sample=True,
                    min_length=10,
                    max_length=10,
                    top_p=1.0,
                    temperature=0.8,
                )
                inp = real_model.generate(
                    inp,
                    do_sample=True,
                    min_length=seqlen,
                    max_length=seqlen,
                    top_p=0.9,
                    temperature=0.8,
                )
            if "glm" in model:
                if hasattr(tokenizer, 'gmask_token_id'):
                    inp[:, -2] = tokenizer.gmask_token_id
                else:
                    inp[:, -2] = 150001
                inp[:, -1] = tokenizer.bos_token_id
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

            gen_inp.append(inp.cpu().numpy())
        gen_inp = np.stack(gen_inp, axis=0)

        np.save(data_path, gen_inp)

    valenc = None
    return trainloader, valenc

def get_ptb_new(nsamples, seed, seqlen, model):
    # from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model):
    # from datasets import load_dataset, load_from_disk
    # traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    traindata = load_from_disk('datasets/allenai/c4/train')
    valdata = load_from_disk('datasets/allenai/c4/validation')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model='', real_model=None, gen_data=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)
    if 'rand_gen' in name:
        return get_random_generalize_tokens(nsamples, seed, seqlen, model, real_model, gen_data=gen_data)
    if 'prompt' in name:
        return get_prompt_tokens(nsamples, seed, seqlen, model, real_model)
