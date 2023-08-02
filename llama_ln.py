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
from data import LLaMaLambadaDataset
from evaluator import LLaMaLambadaEvaluator
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm


def get_llama(model):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = 2048
    return model

class MoveModule(nn.Module):

    def __init__(self, module, dev):
        super().__init__()
        self.module = module
        self.dev = dev

    def forward(self, *inp, **kwargs):
        inp = list(inp)
        ori_dev = inp[0].device
        if inp[0].device != self.dev:
            inp[0] = inp[0].to(self.dev)
        # if cache['mask'] is None or cache['mask'].device != self.dev:
        #     cache['mask'] = kwargs['attention_mask'].to(self.dev)
        # kwargs['attention_mask'] = cache['mask']
        tmp = self.module(*inp, **kwargs)
        return tmp.to(ori_dev)

@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
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
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

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
        full = find_layers(layer)
        if args.true_sequential:
            sequential = [['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'], ['self_attn.o_proj'], ['mlp.up_proj', 'mlp.gate_proj'], ['mlp.down_proj']]
        else:
            sequential = [list(full.keys())]

        norm_layers = find_layers(layer, layers=[LlamaRMSNorm])
        # norm_layers = {k:norm_layers[k] for k in norm_layers if 'input_layernorm' not in k}

        # ========= save float outputs of norm_layers  =========
        # norm_handles = []
        # norm_outs = {}
        # # fp_outs = torch.zeros_like(inps)
        # def add_batch(name):
        #         def tmp(_, inp, out):
        #             # mean = torch.mean(out.view(-1, out.shape[-1]), dim=0)
        #             # std = torch.sqrt(torch.var(out.view(-1, out.shape[-1]), dim=0) + 1e-6)
        #             # norm_outs[name].append([mean, std])
        #             norm_outs[name].append(out)
        #         return tmp
        # for name in norm_layers:
        #     norm_outs[name] = []
        #     norm_handles.append(norm_layers[name].register_forward_hook(add_batch(name)))
        # for j in range(args.nsamples):
        #     _ = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        # for h in norm_handles:
        #     h.remove()

        # ========= save float outputs of attention layers  =========
        ori_outs = []
        for j in range(args.nsamples):
            out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            ori_outs.append(out)

        if len(gpus) > 1:
            layer.self_attn.q_proj = MoveModule(layer.self_attn.q_proj.to(gpus[0]), dev=gpus[0])
            layer.self_attn.k_proj = MoveModule(layer.self_attn.k_proj.to(gpus[1]), dev=gpus[1])
            layer.self_attn.v_proj = MoveModule(layer.self_attn.v_proj.to(gpus[2]), dev=gpus[2])
            layer.self_attn.o_proj = MoveModule(layer.self_attn.o_proj.to(gpus[3]), dev=gpus[3])
            layer.mlp.gate_proj = MoveModule(layer.mlp.gate_proj.to(gpus[4]), dev=gpus[4])
            layer.mlp.down_proj = MoveModule(layer.mlp.down_proj.to(gpus[5]), dev=gpus[5])
            layer.mlp.up_proj = MoveModule(layer.mlp.up_proj.to(gpus[6]), dev=gpus[6])
            layer.input_layernorm = MoveModule(layer.input_layernorm.to(gpus[7]), dev=gpus[7])
            layer.post_attention_layernorm = MoveModule(layer.post_attention_layernorm.to(gpus[7]), dev=gpus[7])
        # for name in full:
        #     full[name] = MoveModule(full[name].to(gpus[gpu_id]), dev=gpus[gpu_id])
        #     gpu_id += 1
        # for name in norm_layers:
        #     norm_layers[name] = MoveModule(norm_layers[name].to(gpus[-1]), dev=gpus[-1])
        
        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
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
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                scale, zero, g_idx, error = gptq[name].fasterquant(blocksize=args.blocksize, percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)

                quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)

                if args.observe:
                    observer.submit(name=name, layerid=i, gptq=gptq[name], error=error)
                else:
                    gptq[name].free()

        # # ========= Optimize LN layers  =========
        norm_params = []
        for name in norm_layers:
            norm_layers[name].is_training = True
            for param in norm_layers[name].parameters():
                # param.requires_grad = True
                param.requires_grad_()
                norm_params.append(param)
        
        iters = 1

        # opt = torch.optim.AdamW(norm_params, lr=1e-3, betas=(0.9, 0.999))
        opt = torch.optim.Adam(norm_params, lr=1e-3)
        # opt = torch.optim.SGD(norm_params, lr=1e-3)
        sche = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=1e-6)
        # loss_func = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        loss_func = torch.nn.MSELoss(reduction='sum')

        # quant_norm_inps = {}
        # quant_norm_handles = []
        # def add_batch(name):
        #         def tmp(_, inp, out):
        #             quant_norm_inps[name].append(inp)
        #         return tmp
        # for name in norm_layers:
        #     quant_norm_inps[name] = []
        #     quant_norm_handles.append(norm_layers[name].register_forward_hook(add_batch(name)))
        # for j in range(args.nsamples):
        #     _ = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        # for h in quant_norm_handles:
        #     h.remove()

        batch_size = 1

        # cur_norm_outs = {}
        # norm_handles = []
        # def add_batch(name):
        #         def tmp(_, inp, out):
        #             cur_norm_outs[name] = out
        #         return tmp
        # for name in norm_layers:
        #     norm_handles.append(norm_layers[name].register_forward_hook(add_batch(name)))

        # layer.train()
        T = 1.0

        # Update Single Norm Layers =============================================================================
        # for name in norm_layers:
        #     norm_layer = norm_layers[name]
        #     norm_layer.train().float()
        #     norm_inp = torch.cat([quant_norm_inps[name][t][0].float() for t in range(args.nsamples)])
        #     norm_out = torch.cat([norm_outs[name][t].float() for t in range(args.nsamples)])
        #     with torch.set_grad_enabled(True):
        #         for it in range(iters):
        #             for j in range(args.nsamples // batch_size):
        #                 opt.zero_grad()
        #                 total_loss = 0
        #                 cur_out = norm_layer(norm_inp[j*batch_size : (j+1)*batch_size])
        #                 # MSE Loss
        #                 # total_loss += loss_func(cur_out, norm_out[j*batch_size : (j+1)*batch_size])
        #                 # mean-std Loss
        #                 tea_mean = torch.mean(norm_out[j*batch_size : (j+1)*batch_size].view(-1, cur_out.shape[-1]), dim=0)
        #                 tea_std = torch.sqrt(torch.var(norm_out[j*batch_size : (j+1)*batch_size].view(-1, cur_out.shape[-1]), dim=0) + 1e-6)

        #                 tmp_mean = torch.mean(cur_out.view(-1, cur_out.shape[-1]), dim=0)
        #                 tmp_std = torch.sqrt(torch.var(cur_out.view(-1, cur_out.shape[-1]), dim=0) + 1e-6)
        #                 total_loss += loss_func(tmp_mean, tea_mean)
        #                 total_loss += loss_func(tmp_std, tea_std)

        #                 # KL-div Loss
        #                 # total_loss += loss_func(F.log_softmax(cur_out / T, dim=-1), 
        #                 #                         F.log_softmax(norm_outs[name][j].detach() / T, dim=-1)) * (T * T)
        #                 total_loss.backward(retain_graph=True)
        #                 # nn.utils.clip_grad_value_(norm_params, clip_value=1.0)
        #                 opt.step()
        #             sche.step()
        #             if it % 1 == 0:
        #                 print("|| Iter: {}, lr: {}, Norm Loss: {}".format(it, opt.param_groups[0]['lr'], total_loss))
        #     norm_layer.eval().half()
        # =========================================================================================================

        # Update All Norm Layers from the whole Attention Layers
        # for name in norm_layers:
        #     norm_layer = norm_layers[name]
        #     norm_layer.train().float()
        # norm_inp = torch.cat([quant_norm_inps[name][t][0].float() for t in range(args.nsamples)])
        # norm_out = torch.cat([norm_outs[name][t].float() for t in range(args.nsamples)])

        batch_inps = torch.cat([inps[t].unsqueeze(0).float() for t in range(args.nsamples)])
        batch_outs = torch.cat([ori_outs[t].float() for t in range(args.nsamples)])
        layer.train().float()
        with torch.set_grad_enabled(True):
            for it in range(iters):
                for j in range(args.nsamples // batch_size):
                    opt.zero_grad()
                    total_loss = 0
                    cur_out = layer(batch_inps[j*batch_size : (j+1)*batch_size], 
                                    attention_mask=torch.stack([attention_mask[0]]*batch_size,dim=0), 
                                    position_ids=torch.stack([position_ids[0]]*batch_size,dim=0))[0]

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
        layer.eval().half()
        # layer.eval()
        # for h in norm_handles:
        #     h.remove()
        for name in norm_layers:
            norm_layers[name].is_training = True
            for param in norm_layers[name].parameters():
                param.requires_grad = False
        
        # ========= End =========================================

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        if len(gpus) > 1:
            layer.self_attn.q_proj = layer.self_attn.q_proj.module
            layer.self_attn.k_proj = layer.self_attn.k_proj.module
            layer.self_attn.v_proj = layer.self_attn.v_proj.module
            layer.self_attn.o_proj = layer.self_attn.o_proj.module
            layer.mlp.gate_proj = layer.mlp.gate_proj.module
            layer.mlp.down_proj = layer.mlp.down_proj.module
            layer.mlp.up_proj = layer.mlp.up_proj.module
            layer.input_layernorm = layer.input_layernorm.module
            layer.post_attention_layernorm = layer.post_attention_layernorm.module

        layers[i] = layer.cpu()
        del layer
        del gptq
        del ori_outs
        del batch_inps
        del batch_outs
        # del quant_norm_inps
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        print('+------------------+--------------+------------+-----------+-------+')
        print('\n')

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
                quantizers['model.layers.%d.%s' % (layerid, name)] = (gptq.quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), wbits, groupsize)

            print(table.draw())
            print('\n')
            gptq.layer.to('cpu')
            gptq.free()

    model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
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
            cache['position_ids'] = kwargs['position_ids']
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
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

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
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
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
def llama_pack(model, quantizers, wbits, groupsize):
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
    from transformers import LlamaConfig, LlamaForCausalLM, modeling_utils
    config = LlamaConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
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

    layers = model.model.layers
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

    for i, layer in enumerate(model.model.layers):
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

    parser.add_argument('model', type=str, help='llama model to load')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'], help='Where to extract calibration data from.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--nearest', action='store_true', help='Whether to run the RTN baseline.')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--trits', action='store_true', help='Whether to use trits for quantization.')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--blocksize', type=int, default=128, help='Blocksize to use for update quantized weights; default as 128.')
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

    args = parser.parse_args()

    if args.layers_dist:
        gpu_dist = [int(x) for x in args.layers_dist.split(':')]
    else:
        gpu_dist = []

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize)
    elif args.load_hf_model:
        model = LlamaForCausalLM.from_pretrained(args.load_hf_model, torch_dtype=torch.float16, device_map='auto')
        model.seqlen = 2048
    else:
        model = get_llama(args.model)
        model.eval()

    dataloader, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen)

    if not args.load and args.wbits < 16 and not args.nearest and not args.load_hf_model:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV)
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
            dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
            print(dataset)
            llama_eval(model, testloader, DEV)

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

    # if not args.observe and args.save:
    if args.save:
        llama_pack(model, quantizers, args.wbits, args.groupsize)
        torch.save(model.state_dict(), args.save)

    if not args.observe and args.save_safetensors:
        llama_pack(model, quantizers, args.wbits, args.groupsize)
        from safetensors.torch import save_file as safe_save
        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        safe_save(state_dict, args.save_safetensors)

    if args.data_path is not None:
        from transformers import LlamaTokenizer, LlamaForCausalLM
        tokenizer = LlamaTokenizer.from_pretrained(args.model)
        tokenizer.pad_token = tokenizer.eos_token
        dataset = LLaMaLambadaDataset(args.data_path, tokenizer)
        evaluator = LLaMaLambadaEvaluator(dataset, tokenizer, 'cuda')

        # model_fp16 = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map='auto')
        # acc_fp16 = evaluator.evaluate(model_fp16.to(DEV))
        # print(f'Original model (fp16) accuracy: {acc_fp16}')

        tick = time.time()
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            acc_quant = evaluator.evaluate(model)
        else:
            acc_quant = evaluator.evaluate(model.to(DEV))
        print('Quantized model accuracy: {:0.4f}'.format(acc_quant))
        print(time.time() - tick)
