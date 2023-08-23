import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import quant

from gptq import GPTQ, Observer
from utils import find_layers, DEV, set_seed, get_wikitext2, get_ptb, get_c4, get_ptb_new, get_c4_new, get_loaders, export_quant_table, gen_conditions
from texttable import Texttable
import transformers
from data import LambadaDataset
from evaluator import LambadaEvaluator
from transformers import AutoTokenizer, BloomTokenizerFast, BloomForCausalLM
from torch.nn import LayerNorm
from llama_ln import MoveModule

def get_bloom(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import BloomForCausalLM
    model = BloomForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = 2048
    return model


@torch.no_grad()
def bloom_sequential(model, dataloader, dev, means=None, stds=None, update_norm=False, lr=1e-6):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None, 'alibi': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['alibi'] = kwargs['alibi']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    alibi = cache['alibi']

    print('Ready.')

    quantizers = {}
    observer = Observer()

    gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
    for i in range(len(layers)):

        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+------------------+--------------+------------+-----------+-------+')
        print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
        print('+==================+==============+============+===========+=======+')

        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gptq = {}

        if update_norm:
            norm_layers = find_layers(layer, layers=[LayerNorm])
            # gpu_id = 0
            if len(gpus) > 1:
                layer.self_attention.query_key_value = MoveModule(layer.self_attention.query_key_value.to(gpus[0]), dev=gpus[0])
                layer.self_attention.dense = MoveModule(layer.self_attention.dense.to(gpus[1]), dev=gpus[1])
                layer.mlp.dense_h_to_4h = MoveModule(layer.mlp.dense_h_to_4h.to(gpus[2]), dev=gpus[2])
                layer.mlp.dense_4h_to_h = MoveModule(layer.mlp.dense_4h_to_h.to(gpus[3]), dev=gpus[3])
                # layer.mlp.gate_proj = MoveModule(layer.mlp.gate_proj.to(gpus[4]), dev=gpus[4])
                # layer.mlp.down_proj = MoveModule(layer.mlp.down_proj.to(gpus[5]), dev=gpus[5])
                # layer.mlp.up_proj = MoveModule(layer.mlp.up_proj.to(gpus[6]), dev=gpus[6])
                layer.input_layernorm = MoveModule(layer.input_layernorm.to(gpus[3]), dev=gpus[3])
                layer.post_attention_layernorm = MoveModule(layer.post_attention_layernorm.to(gpus[3]), dev=gpus[3])

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
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
        for h in handles:
            h.remove()

        for name in subset:
            scale, zero, g_idx, error = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
            quantizers['transformer.h.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)

            if args.observe:
                observer.submit(name=name, layerid=i, gptq=gptq[name], error=error)
            else:
                gptq[name].free()

        # # ========= Optimize LN layers  =========
        if update_norm:
            norm_params = []
            for name in norm_layers:
                norm_layers[name].is_training = True
                for param in norm_layers[name].parameters():
                    # param.requires_grad = True
                    param.requires_grad_()
                    norm_params.append(param)
            
            iters = 1

            # opt = torch.optim.AdamW(norm_params, lr=1e-3, betas=(0.9, 0.999))
            opt = torch.optim.Adam(norm_params, lr=lr)
            # opt = torch.optim.SGD(norm_params, lr=1e-3)
            sche = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr)
            # loss_func = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
            # loss_func = torch.nn.MSELoss(reduction='sum')
            logsoftmax = torch.nn.LogSoftmax(dim=1)
            batch_size = 1
            T = 1.0

            batch_inps = torch.cat([inps[t].unsqueeze(0).float() for t in range(args.nsamples)])
            batch_outs = torch.cat([outs[t].unsqueeze(0).float() for t in range(args.nsamples)])
            layer.train().float()
            with torch.set_grad_enabled(True):
                for it in range(iters):
                    for j in range(args.nsamples // batch_size):
                        opt.zero_grad()
                        total_loss = 0
                        cur_out = layer(batch_inps[j*batch_size : (j+1)*batch_size],
                                        attention_mask=torch.stack([attention_mask[0]]*batch_size,dim=0),
                                        alibi=torch.stack([alibi[0]]*batch_size,dim=0).float())[0]
                        
                        # CWD Loss
                        B, S, N = cur_out.shape
                        softmax_pred_T = F.softmax(batch_outs[j*batch_size : (j+1)*batch_size].transpose(2, 1).view(-1, S) / T, dim=1)

                        loss = torch.sum(softmax_pred_T *
                                        logsoftmax(batch_outs[j*batch_size : (j+1)*batch_size].transpose(2, 1).view(-1, S) / T) -
                                        softmax_pred_T *
                                        logsoftmax(cur_out.transpose(2, 1).view(-1, S) / T)) * (
                                            T**2)

                        total_loss = loss / (B * N)
                        # MSE Loss
                        # total_loss += loss_func(cur_out, batch_outs[j*batch_size : (j+1)*batch_size])

                        # mean-std Loss
                        # tea_mean = torch.mean(batch_outs[j*batch_size : (j+1)*batch_size].view(-1, cur_out.shape[-1]), dim=0)
                        # tea_std = torch.sqrt(torch.var(batch_outs[j*batch_size : (j+1)*batch_size].view(-1, cur_out.shape[-1]), dim=0) + 1e-6)

                        # tmp_mean = torch.mean(cur_out.view(-1, cur_out.shape[-1]), dim=0)
                        # tmp_std = torch.sqrt(torch.var(cur_out.view(-1, cur_out.shape[-1]), dim=0) + 1e-6)
                        # total_loss += loss_func(tmp_mean, tea_mean)
                        # total_loss += loss_func(tmp_std, tea_std)

                        # KL-div Loss
                        # total_loss += loss_func(F.log_softmax(cur_out / T, dim=-1), 
                        #                         F.log_softmax(norm_outs[name][j].detach() / T, dim=-1)) * (T * T)
                        total_loss.backward()
                        # nn.utils.clip_grad_value_(norm_params, clip_value=1.0)
                        opt.step()
                    sche.step()
                    if it % 1 == 0:
                        print("|| Iter: {}, lr: {}, Norm Loss: {}".format(it, opt.param_groups[0]['lr'], total_loss))
            layer.eval().half()
            # layer.eval()
            # for h in norm_handles:
            #     h.remove()
            for name in norm_layers:
                norm_layers[name].is_training = True
                for param in norm_layers[name].parameters():
                    param.requires_grad = False

            if len(gpus) > 1:
                layer.self_attention.query_key_value = layer.self_attention.query_key_value.module
                layer.self_attention.dense = layer.self_attention.dense.module
                layer.mlp.dense_h_to_4h = layer.mlp.dense_h_to_4h.module
                layer.mlp.dense_4h_to_h = layer.mlp.dense_4h_to_h.module
                layer.input_layernorm = layer.input_layernorm.module
                layer.post_attention_layernorm = layer.post_attention_layernorm.module
            
            del batch_inps
            del batch_outs
        # ========= End =========================================
        layer = layer.to(dev)
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        print('+------------------+--------------+------------+-----------+-------+')
        print('\n')

    # record_handles = []
    # quant_outs = {}
    # def record_dist_quant(idx):
    #         def tmp(_, inp, out):
    #             _out = out[0]
    #             mean = torch.mean(_out.view(-1, _out.shape[-1]), dim=0).cpu().numpy()
    #             std = torch.sqrt(torch.var(_out.view(-1, _out.shape[-1]), dim=0) + 1e-6).cpu().numpy()
    #             quant_outs[idx].append([mean, std])
    #         return tmp
    # for i in range(len(layers)):
    #     quant_outs[i] = []
    #     record_handles.append(layers[i].register_forward_hook(record_dist_quant(i)))

    # model = model.to(dev)
    # for batch in dataloader:
    #     model(batch[0].to(dev))

    # for i in range(len(quant_outs)):
    #     np.save("weights_dist/bloom_output_q_ln_mean2/layer_{}".format(i), np.array(quant_outs[i]))
    # del quant_outs
    # for h in record_handles:
    #     h.remove()

    if args.observe:
        observer.print()
        conditions = gen_conditions(args.wbits, args.groupsize)
        for item in observer.items():
            name = item[0]
            layerid = item[1]
            gptq = item[2]['gptq']
            error = item[2]['error']
            target = error / 2

            table = Texttable()
            table.header(['wbits', 'groupsize', 'error'])
            table.set_cols_dtype(['i', 'i', 'f'])
            table.add_row([args.wbits, args.groupsize, error])

            print('Optimizing {} {} ..'.format(name, layerid))
            for wbits, groupsize in conditions:

                if error < target:
                    # if error dropped 50%, skip
                    break

                gptq.quantizer.configure(wbits, perchannel=True, sym=args.sym, mse=False)

                scale, zero, g_idx, error = gptq.fasterquant(percdamp=args.percdamp, groupsize=groupsize, actorder=args.act_order, name=name)

                table.add_row([wbits, groupsize, error])
                quantizers['transformer.h.%d.%s' % (layerid, name)] = (gptq.quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), wbits, groupsize)

            print(table.draw())
            print('\n')
            gptq.layer.to('cpu')
            gptq.free()

    model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def bloom_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None, 'alibi': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['alibi'] = kwargs['alibi']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    alibi = cache['alibi']

    for i in range(len(layers)):
        # print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = quant.Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantizer.quantize(W).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
        layers[i] = layer.cpu() 
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.transformer.ln_f = model.transformer.ln_f.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        hidden_states = model.transformer.ln_f(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


# TODO: perform packing on GPU
def bloom_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    quant.make_quant_linear(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [quant.QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
    return model


def load_quant(model, checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
    from transformers import BloomConfig, BloomForCausalLM, modeling_utils
    config = BloomConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = BloomForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)

    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)
    model.seqlen = 2048
    print('Done.')

    return model


def llama_multigpu(model, gpus, gpu_dist):
    model.model.embed_tokens = model.model.embed_tokens.to(gpus[0])
    if hasattr(model.model, 'norm') and model.model.norm:
        model.model.norm = model.model.norm.to(gpus[-1])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])

    cache = {'mask': None}

    class MoveModule(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device

        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if cache['mask'] is None or cache['mask'].device != self.dev:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.transformer.h
    from math import ceil
    if not gpu_dist:
        pergpu = ceil(len(layers) / len(gpus))
        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))
    else:
        assigned_gpus = []
        for i in range(len(gpu_dist)):
            assigned_gpus = assigned_gpus + [i] * gpu_dist[i]

        remaining_assignments = len(layers)-len(assigned_gpus)
        if remaining_assignments > 0:
            assigned_gpus = assigned_gpus + [-1] * remaining_assignments

        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(gpus[assigned_gpus[i]]))

    model.gpus = gpus


def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

    cache = {'past': None}

    def clear_past(i):

        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None

        return tmp

    for i, layer in enumerate(model.transformer.h):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(input_ids[:, i:i + 1], past_key_values=cache['past'], attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1)))
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if hasattr(model, 'gpus'):
                mem_allocated = sum(torch.cuda.memory_allocated(gpu) for gpu in model.gpus) / 1024 / 1024
            else:
                mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            max_memory = max(max_memory, mem_allocated)
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())
            print('max memory(MiB):', max_memory)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='BLOOM model to load; pass `bigscience/bloom-X`.')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4', 'rand_gen'], help='Where to extract calibration data from.')
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
    parser.add_argument("--save_unpack", type=str, default='')
    parser.add_argument("--save_hf_model", type=str, default='')
    parser.add_argument("--load_hf_model", type=str, default='')
    parser.add_argument('--update_norm', action='store_true', help='Whether to use the new PTB and C4 eval')
    parser.add_argument('--lr', type=float, default=1.e-6, help='Learning Rate for Norm Tuning.')
    parser.add_argument('--gen_data', type=str, help='BLOOM model to load; pass `bigscience/bloom-X`.')

    args = parser.parse_args()

    if args.layers_dist:
        gpu_dist = [int(x) for x in args.layers_dist.split(':')]
    else:
        gpu_dist = []

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize)
        model = model.to(DEV)
    elif args.load_hf_model:
        model = BloomForCausalLM.from_pretrained(args.load_hf_model, torch_dtype=torch.float16, device_map='auto')
        model.seqlen = 2048
    else:
        model = get_bloom(args.model)
        model.eval()

    dataloader, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, 
            model=args.model, seqlen=model.seqlen, 
            # real_model=model.to(DEV), gen_data=args.gen_data
            )

    if not args.load and not args.load_hf_model and args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = bloom_sequential(model, dataloader, DEV, update_norm=args.update_norm, lr=args.lr)
        print(time.time() - tick)

    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            benchmark(model, input_ids, check=args.check)

    if args.eval:
        datasets = ['wikitext2', 'ptb', 'c4']
        if args.new_eval:
            datasets = ['wikitext2', 'ptb-new', 'c4-new']
        for dataset in datasets: 
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print(dataset)
            bloom_eval(model, testloader, DEV)

    if args.quant_directory is not None:
        export_quant_table(quantizers, args.quant_directory)

    if args.save_hf_model:
        if not os.path.exists(args.save_hf_model):
            os.makedirs(args.save_hf_model)
        quantizers_dict = quantizers.copy()
        for key, value in quantizers_dict.items():
            quantizers_dict[key] = quantizers_dict[key][1:]
        torch.save(quantizers_dict, os.path.join(args.save_hf_model, 'quantizers.pt'))
        model.save_pretrained(args.save_hf_model)
    
    if args.save_unpack:
        torch.save(model.state_dict(), args.save_unpack)
    
    if args.save:
        bloom_pack(model, quantizers, args.wbits, args.groupsize)
        torch.save(model.state_dict(), args.save)

    if not args.observe and args.save_safetensors:
        bloom_pack(model, quantizers, args.wbits, args.groupsize)
        from safetensors.torch import save_file as safe_save
        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        safe_save(state_dict, args.save_safetensors)

    if args.data_path is not None:
        from transformers import AutoTokenizer, BloomTokenizerFast, BloomForCausalLM
        tokenizer = BloomTokenizerFast.from_pretrained(args.model, padding_side='left')
        dataset = LambadaDataset(args.data_path, tokenizer)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1)
        evaluator = LambadaEvaluator(data_loader, tokenizer, 'cuda')

        # model_fp16 = BloomForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map='auto')
        # acc_fp16 = evaluator.evaluate(model_fp16)
        # print(f'Original model (fp16) accuracy: {acc_fp16}')

        tick = time.time()
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            acc_quant = evaluator.evaluate(model)
        else:
            acc_quant = evaluator.evaluate(model.to(DEV))
        print('Quantized model accuracy: {:0.4f}'.format(acc_quant))
        print(time.time() - tick)
