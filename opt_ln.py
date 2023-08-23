import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

import transformers
from gptq import GPTQ
from utils import find_layers, DEV, set_seed, get_wikitext2, get_ptb, get_c4, get_ptb_new, get_c4_new, get_loaders
import quant
from data import OPTLambadaDataset
from evaluator import LLaMaLambadaEvaluator
from transformers.models.opt.modeling_opt import OPTForCausalLM
from torch.nn import LayerNorm
from llama_ln import MoveModule


def get_opt(model):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    OPTForCausalLM._keys_to_ignore_on_load_missing = [r"dense_alpha", r"fc2_alpha"]
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model


@torch.no_grad()
def opt_sequential(model, dataloader, dev, update_norm=False):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    quantizers = {}
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
            layer.self_attn.k_proj = MoveModule(layer.self_attn.k_proj.to(gpus[1]), dev=gpus[1])
            layer.self_attn.v_proj = MoveModule(layer.self_attn.v_proj.to(gpus[1]), dev=gpus[1])
            layer.self_attn.q_proj = MoveModule(layer.self_attn.q_proj.to(gpus[2]), dev=gpus[2])
            layer.self_attn.out_proj = MoveModule(layer.self_attn.out_proj.to(gpus[2]), dev=gpus[2])
            layer.fc1 = MoveModule(layer.fc1.to(gpus[3]), dev=gpus[3])
            layer.fc2 = MoveModule(layer.fc2.to(gpus[3]), dev=gpus[3])
            layer.self_attn_layer_norm = MoveModule(layer.self_attn_layer_norm.to(gpus[3]), dev=gpus[3])
            layer.final_layer_norm = MoveModule(layer.final_layer_norm.to(gpus[3]), dev=gpus[3])

        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = quant.Quantizer()
            gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False, trits=args.trits)

        def add_batch(name):

            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()

        for name in subset:
            # print(f'Quantizing {name} in layer {i+1}/{len(layers)}...')
            scale, zero, g_idx, _ = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
            quantizers['model.decoder.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu())
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
            opt = torch.optim.Adam(norm_params, lr=5e-5)
            # opt = torch.optim.SGD(norm_params, lr=1e-3)
            sche = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=1e-8)
            # loss_func = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
            loss_func = torch.nn.MSELoss(reduction='sum')
            batch_size = 1
            T = 1.0

            batch_inps = torch.cat([inps[t].unsqueeze(0) for t in range(args.nsamples)]).float()
            batch_outs = torch.cat([outs[t].unsqueeze(0) for t in range(args.nsamples)]).float()
            layer.train().float()
            with torch.set_grad_enabled(True):
                for it in range(iters):
                    for j in range(args.nsamples // batch_size):
                        opt.zero_grad()
                        total_loss = 0
                        cur_out = layer(batch_inps[j*batch_size : (j+1)*batch_size],
                                        attention_mask=attention_mask)[0]

                        # MSE Loss
                        total_loss += loss_func(cur_out, batch_outs[j*batch_size : (j+1)*batch_size])

                        # mean-std Loss
                        # tea_mean = torch.mean(norm_out[j*batch_size : (j+1)*batch_size].view(-1, cur_out.shape[-1]), dim=0)
                        # tea_std = torch.sqrt(torch.var(norm_out[j*batch_size : (j+1)*batch_size].view(-1, cur_out.shape[-1]), dim=0) + 1e-6)

                        # tmp_mean = torch.mean(cur_out.view(-1, cur_out.shape[-1]), dim=0)
                        # tmp_std = torch.sqrt(torch.var(cur_out.view(-1, cur_out.shape[-1]), dim=0) + 1e-6)
                        # total_loss += loss_func(tmp_mean, tea_mean)
                        # total_loss += loss_func(tmp_std, tea_std)

                        # KL-div Loss
                        # total_loss += loss_func(F.log_softmax(cur_out / T, dim=-1), 
                        #                         F.log_softmax(norm_outs[name][j].detach() / T, dim=-1)) * (T * T)
                        total_loss.backward(retain_graph=True)
                        # nn.utils.clip_grad_value_(norm_params, clip_value=1.0)
                        opt.step()
                    sche.step()
                    if it % 1 == 0:
                        print("|| Iter: {}, lr: {}, Norm Loss: {}".format(it, opt.param_groups[0]['lr'], total_loss))

            layer.self_attn.k_proj = layer.self_attn.k_proj.module
            layer.self_attn.v_proj = layer.self_attn.v_proj.module
            layer.self_attn.q_proj = layer.self_attn.q_proj.module
            layer.self_attn.out_proj = layer.self_attn.out_proj.module
            layer.fc1 = layer.fc1.module
            layer.fc2 = layer.fc2.module
            layer.self_attn_layer_norm = layer.self_attn_layer_norm.module
            layer.final_layer_norm = layer.final_layer_norm.module

            layer.eval().half()
            for name in norm_layers:
                norm_layers[name].is_training = True
                for param in norm_layers[name].parameters():
                    param.requires_grad = False
            
            del batch_inps
            del batch_outs
        # ========= End =========================================
        layer = layer.to(dev)

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        print('+------------------+--------------+------------+-----------+-------+')
        print('\n')

    model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def opt_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
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
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
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
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
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
def opt_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    quant.make_quant_linear(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [quant.QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
    return model


def load_quant(model, checkpoint, wbits, groupsize=-1, eval=True, warmup_autotune=True):
    from transformers import OPTConfig, OPTForCausalLM
    config = OPTConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = OPTForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['model.decoder.project_out', 'model.decoder.project_in', 'lm_head']:
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

    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
    model.seqlen = model.config.max_position_embeddings
    print('Done.')
    return model


def opt_multigpu(model, gpus):
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(gpus[0])
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(gpus[0])
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(gpus[0])
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(gpus[-1])
    if hasattr(model.model.decoder, 'final_layer_norm') and model.model.decoder.final_layer_norm:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(gpus[-1])
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

    layers = model.model.decoder.layers
    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers)):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))

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

    for i, layer in enumerate(model.model.decoder.layers):
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

    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(input_ids[:, i].reshape(-1), past_key_values=cache['past'], attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1)))
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        import numpy as np
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='OPT model to load; pass `facebook/opt-X`.')
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
    parser.add_argument('--new-eval', action='store_true', help='Whether to use the new PTB and C4 eval')
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--save_hf_model", type=str, default='')
    parser.add_argument("--load_hf_model", type=str, default='')
    parser.add_argument('--update_norm', action='store_true', help='Whether to use the new PTB and C4 eval')

    args = parser.parse_args()

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize)
        model = model.to(DEV)
    elif args.load_hf_model:
        model = OPTForCausalLM.from_pretrained(args.load_hf_model, torch_dtype=torch.float16, device_map='auto')
        model.seqlen = model.config.max_position_embeddings
    else:
        model = get_opt(args.model)
        model.eval()

    dataloader, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen)

    if not args.load and not args.load_hf_model and args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = opt_sequential(model, dataloader, DEV, update_norm=args.update_norm)
        print("|| GPTQ Time ===================================================")
        print(time.time() - tick)

    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            opt_multigpu(model, gpus)
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
            dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
            print(dataset)
            opt_eval(model, testloader, DEV)

    if args.save_hf_model:
        if not os.path.exists(args.save_hf_model):
            os.makedirs(args.save_hf_model)
        quantizers_dict = quantizers.copy()
        for key, value in quantizers_dict.items():
            quantizers_dict[key] = quantizers_dict[key][1:]
        torch.save(quantizers_dict, os.path.join(args.save_hf_model, 'quantizers.pt'))
        model.save_pretrained(args.save_hf_model)
    
    if args.save:
        opt_pack(model, quantizers, args.wbits, args.groupsize)
        torch.save(model.state_dict(), args.save)

    if args.save_safetensors:
        opt_pack(model, quantizers, args.wbits, args.groupsize)
        from safetensors.torch import save_file as safe_save
        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        safe_save(state_dict, args.save_safetensors)

    if args.data_path is not None:
        from transformers.models.opt.modeling_opt import OPTForCausalLM
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(args.model, padding_side='left')
        dataset = OPTLambadaDataset(args.data_path, tokenizer)
        evaluator = LLaMaLambadaEvaluator(dataset, tokenizer, 'cuda')

        # OPTForCausalLM._keys_to_ignore_on_load_missing = [r"dense_alpha", r"fc2_alpha"]
        # model_fp16 = OPTForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map='auto')
        # acc_fp16 = evaluator.evaluate(model_fp16)
        # print(f'Original model (fp16) accuracy: {acc_fp16}')

        tick = time.time()
        acc_quant = evaluator.evaluate(model.to(DEV))
        print('Quantized model accuracy: {:0.4f}'.format(acc_quant))
        print(time.time() - tick)
