import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import quant
import copy

from gptq import GPTQ, Observer
from utils import find_layers, DEV, set_seed, get_wikitext2, get_ptb, get_c4, get_ptb_new, get_c4_new, get_loaders, export_quant_table, gen_conditions
from texttable import Texttable
import transformers
from data import GLMLambadaDataset
from evaluator import GLMLambadaEvaluator
from transformers import AutoTokenizer, AutoModel


def get_glm(model, device_map):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = AutoModel.from_pretrained(model, torch_dtype=torch.float16, trust_remote_code=True, device_map=device_map)
    model.seqlen = 2048
    return model

@torch.no_grad()
def glm_sense_test(args, model, dataloader, dev, path):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dataset = GLMLambadaDataset(args.data_path, tokenizer)
    evaluator = GLMLambadaEvaluator(dataset, tokenizer, 'cuda')

    print('Starting Sensitivity Testing...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.layers

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, 1, model.config.hidden_size), dtype=dtype, device=model.device)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'layer_id': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache['layer_id'] = kwargs['layer_id']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].cuda())
        except ValueError:
            pass
    layers[0] = layers[0].module

    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    layer_id = cache['layer_id']

    print('Ready.')

    sense_scores = []
    for sense_layer in range(len(layers)):
        for i in range(sense_layer+1):

            print(f'Processing layer {i+1}/{len(layers)}..')
            print('+------------------+--------------+------------+-----------+-------+')
            print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
            print('+==================+==============+============+===========+=======+')

            layer_orin = copy.deepcopy(layers[i])  # 量化前的层
            layer = layers[i]
            subset = find_layers(layer)  # 要量化的当前层
            gptq = {}

            if i == sense_layer:
                for name in subset:
                    gptq[name] = GPTQ(subset[name], observe=args.observe)
                    gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j], position_ids=position_ids, attention_mask=attention_mask, layer_id=layer_id)[0]
            for h in handles:
                h.remove()

            if i == sense_layer:
                for name in subset:
                    scale, zero, g_idx, error = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                    # quantizers['transformer.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)

                    gptq[name].free()
            else:
                print('Current layer is not quantized')

            for j in range(args.nsamples):
                outs[j] = layer(inps[j], position_ids=position_ids, attention_mask=attention_mask, layer_id=layer_id)[0]

            if i == sense_layer:
                layers[i] = layer
            del layer
            del gptq
            torch.cuda.empty_cache()
            inps, outs = outs, inps
            print('+------------------+--------------+------------+-----------+-------+')
            print('\n')

        model.config.use_cache = use_cache

        # 测试
        tick = time.time()
        acc_quant = evaluator.evaluate(model)
        print('Quantized model accuracy: {:0.4f}'.format(acc_quant))
        sense_scores.append(acc_quant)
        print(time.time() - tick)

        model.transformer.layers[i] = layer_orin  # 测试结束恢复量化前的权重

    return sense_scores


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='BLOOM model to load; pass `bigscience/bloom-X`.')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'], help='Where to extract calibration data from.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--nearest', action='store_true', help='Whether to run the RTN baseline.')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--trits', action='store_true', help='Whether to use trits for quantization.')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--eval', action='store_true', help='evaluate quantized model.')
    parser.add_argument('--save', type=str, default='', help='Save quantized checkpoint under this name.')
    parser.add_argument('--save_safetensors', type=str, default='', help='Save quantized `.safetensors` checkpoint under this name.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model.')
    parser.add_argument('--benchmark', type=int, default=0, help='Number of tokens to use for benchmarking.')
    parser.add_argument('--check', action='store_true', help='Whether to compute perplexity during benchmarking for verification.')
    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')
    parser.add_argument('--new-eval', action='store_true', help='Whether to use the new PTB and C4 eval')
    parser.add_argument('--layers-dist', type=str, default='', help='Distribution of layers across GPUs. e.g. 2:1:1 for 2 layers on GPU 0, 1 layer on GPU 1, and 1 layer on GPU 2. Any remaining layers will be assigned to your last GPU.')
    parser.add_argument('--observe',
                        action='store_true',
                        help='Auto upgrade layer precision to higher precision, for example int2 to int4, groupsize 128 to 64. \
            When this feature enabled, `--save` or `--save_safetensors` would be disable.')
    parser.add_argument('--quant-directory', type=str, default=None, help='Specify the directory for export quantization parameters to toml format. `None` means no export by default.')
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--save_hf_model", type=str, default='')
    parser.add_argument("--load_hf_model", type=str, default='')
    parser.add_argument("--sense_test", type=str, default='./sense_test/res.txt')

    args = parser.parse_args()

    if not os.path.exists('./sense_test/'):
        os.makedirs('./sense_test/')

    if 'glm-130b' in args.model:
        with open(os.path.join(args.load_hf_model, "device_map.json"), "r") as infile:
            device_map = json.load(infile)
    else:
        device_map = 'auto'
    model = get_glm(args.model, device_map=device_map)
    model.eval()

    dataloader, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen)
    sense_scores = glm_sense_test(args, model, dataloader, DEV, args.sense_test)

    sorted_scores = sorted(enumerate(sense_scores), key=lambda x: x[1], reverse=True)
    sorted_dict = {x[0]: x[1] for x in sorted_scores}
    print(sorted_dict)
    with open(args.sense_test, 'w') as f:
        for key, value in sorted_dict.items():
            f.write(f'{key}: {value}\n')
    print('done.')
