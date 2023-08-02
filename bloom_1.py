import math
import time

import torch
import torch.nn as nn
import transformers

from gptq import * 
from utils.modelutils import *
from quant import *
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
    model = BloomForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

@torch.no_grad()
def bloom_sequential(model, dataloader, dev, means=None, stds=None, update_norm=False, lr=1e-6, rand_inp=False):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
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

    if rand_inp:
        inps = torch.randn_like(inps)
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    alibi = cache['alibi']

    print('Ready.')
    gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        print("[Layer {}] mean: {} | std: {} | min: {} | max: {}".format(i, inps.mean(), inps.std(), inps.min(), inps.max()))
        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+------------------+--------------+------------+-----------+-------+')
        print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
        print('+==================+==============+============+===========+=======+')
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
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False
            )

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
            print(i, name)
            # print('Quantizing ...')
            gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize)
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
            loss_func = torch.nn.MSELoss(reduction='sum')
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

                        # MSE Loss
                        # total_loss += loss_func(cur_out, batch_outs[j*batch_size : (j+1)*batch_size])

                        # mean-std Loss
                        tea_mean = torch.mean(batch_outs[j*batch_size : (j+1)*batch_size].view(-1, cur_out.shape[-1]), dim=0)
                        tea_std = torch.sqrt(torch.var(batch_outs[j*batch_size : (j+1)*batch_size].view(-1, cur_out.shape[-1]), dim=0) + 1e-6)

                        tmp_mean = torch.mean(cur_out.view(-1, cur_out.shape[-1]), dim=0)
                        tmp_std = torch.sqrt(torch.var(cur_out.view(-1, cur_out.shape[-1]), dim=0) + 1e-6)
                        total_loss += loss_func(tmp_mean, tea_mean)
                        total_loss += loss_func(tmp_std, tea_std)

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
        # del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

@torch.no_grad()
def bloom_eval(model, testenc, dev):
    print('Evaluation...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
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
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

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
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


if __name__ == '__main__':
    import argparse
    from utils.datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='BLOOM model to load; pass `bigscience/bloom-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4', 'rand_gen', 'prompt'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval'
    )

    parser.add_argument('--update_norm', action='store_true', help='Whether to use the new PTB and C4 eval')
    parser.add_argument('--random_inp', action='store_true', help='Whether to use the new PTB and C4 eval')
    parser.add_argument('--lr', type=float, default=1.e-6, help='Learning Rate for Norm Tuning.')
    parser.add_argument('--gen_data', type=str, help='BLOOM model to load; pass `bigscience/bloom-X`.')

    args = parser.parse_args()

    model = get_bloom(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen, real_model=model.to(DEV), gen_data=args.gen_data
    )

    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        bloom_sequential(model, dataloader, DEV, update_norm=args.update_norm, lr=args.lr, rand_inp=args.random_inp)
        print(time.time() - tick)

    datasets = ['wikitext2', 'ptb', 'c4']
    if args.new_eval:
      datasets = ['wikitext2', 'ptb-new', 'c4-new']
    for dataset in datasets: 
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        bloom_eval(model, testloader, DEV)