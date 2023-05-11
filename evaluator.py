import torch
from tqdm import tqdm
from typing import Dict, List
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 20005] = 5e4
        return scores


class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in tqdm(self.dataset):
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc

def split_inputs_and_targets(entries: Dict[str, torch.LongTensor],
                             pad_token_id: int,
                             pad_to_left=True):
    input_ids = entries['input_ids']
    attn_mask = entries['attention_mask']
    token_type_ids = entries['token_type_ids']

    # Split inputs and labels by token_type_ids.
    input_token_ids = [
        ids[(mask == 1) & (type_ids == 0)]
        for ids, mask, type_ids in zip(input_ids, attn_mask, token_type_ids)]
    # FT allows int32 tensors.
    input_lengths = torch.tensor(
        [len(input_tokens) for input_tokens in input_token_ids]).int()
    max_length = input_lengths.max()
    input_token_ids = torch.stack([
        torch.nn.functional.pad(
            token_ids,
            pad=[max_length - len(token_ids), 0]
                if pad_to_left else [0, max_length - len(token_ids)],
            mode='constant',
            value=pad_token_id
        ) for token_ids in input_token_ids]).int()
    target_token_ids = [
        ids[(mask == 1) & (type_ids == 1)]
        for ids, mask, type_ids in zip(input_ids, attn_mask, token_type_ids)]
    return input_token_ids, input_lengths, target_token_ids

class LambadaEvaluator:
    def __init__(self, dataloader, tokenizer, device):
        self.data_loader = dataloader
        self.tokenizer = tokenizer
        self.device = device
        self.total_batches = -1

    def set_total_batches(self, total):
        self.total_batches = total

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        num_corrects = 0
        num_requests = 0
        count = 0
        for entries in tqdm(self.data_loader):
            input_token_ids, input_lengths, target_token_ids = \
                split_inputs_and_targets(entries, self.tokenizer.pad_token_id)
            # input_token_ids are already padded
            batch_size = input_token_ids.shape[0]
            output_length = max([len(target) for target in target_token_ids])
            # Outputs (batch_size, seq_length)
            outputs = model.generate(inputs=input_token_ids.cuda(),
                                     max_new_tokens=output_length,
                                     num_beams=1,
                                     temperature=1,
                                     top_k=1,
                                     top_p=0,
                                     repetition_penalty=1.0,
                                     length_penalty=1.0)

            # output_token_ids: padding/input/output
            output_token_ids = outputs[:, input_token_ids.shape[1]:]
            output_token_ids = [
                out[:len(tgt)].cpu()
                for out, tgt in zip(output_token_ids, target_token_ids)]

            # output_texts = self.tokenizer.batch_decode(output_token_ids)
            # target_texts = self.tokenizer.batch_decode(target_token_ids)
            # input_texts = self.tokenizer.batch_decode(input_token_ids)
            # Convert to output objects.
            for i in range(batch_size):
                out = output_token_ids[i]
                tgt = target_token_ids[i].cpu()
                if len(out) == len(tgt):
                    is_correct = (tgt == out).all()
                    num_corrects += int(is_correct)
            num_requests += batch_size
            count += 1
            if self.total_batches > 0 and count >= self.total_batches:
                break

        accuracy = num_corrects * 100 / num_requests
        # Reference: HF model's LAMBADA Accuracy for bloom-560m ~ 35.36%
        print(f'Accuracy: {accuracy:0.4f}% ({num_corrects}/{num_requests}) ')
        return accuracy

class GLMLambadaEvaluator:
    def __init__(self, dataloader, tokenizer, device, bidir=False):
        self.data_loader = dataloader
        self.tokenizer = tokenizer
        self.device = device
        self.total_batches = -1
        self.bidirectional = bidir

    def set_total_batches(self, total):
        self.total_batches = total

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        num_corrects = 0
        num_requests = 0
        count = 0
        for entries in tqdm(self.data_loader):
            input_token_ids, target_token_ids = entries['input_ids'], entries['target_ids']
            if self.bidirectional is True:
                # MASK = 150000 gMASK = 150001
                input_token_ids[:, -2] = 150000
            # input_token_ids are already padded
            batch_size = input_token_ids.shape[0]
            input_length = max([len(input) for input in input_token_ids])
            output_length = max([len(target) for target in target_token_ids])
            logits_processor = LogitsProcessorList()
            logits_processor.append(InvalidScoreLogitsProcessor())
            # Outputs (batch_size, seq_length)
            outputs = model.generate(inputs=input_token_ids.cuda(),
                                     max_new_tokens=output_length,
                                     do_sample=False,
                                     num_beams=1,
                                     temperature=1,
                                     top_k=1,
                                     top_p=0,
                                     logits_processor=logits_processor)

            # output_token_ids: input/padding/output
            output_token_ids = outputs[:, input_token_ids.shape[1]:]
            output_token_ids = [
                out[:len(tgt)].cpu()
                for out, tgt in zip(output_token_ids, target_token_ids)]

            # output_texts = self.tokenizer.batch_decode(output_token_ids)
            # target_texts = self.tokenizer.batch_decode(target_token_ids)
            # input_texts = self.tokenizer.batch_decode(input_token_ids)
            # Convert to output objects.
            for i in range(batch_size):
                out = output_token_ids[i]
                tgt = target_token_ids[i].cpu()
                if len(out) == len(tgt):
                    is_correct = (tgt == out).all()
                    num_corrects += int(is_correct)
            num_requests += batch_size
            count += 1
            if self.total_batches > 0 and count >= self.total_batches:
                break

        accuracy = num_corrects * 100 / num_requests
        # Reference: HF model's LAMBADA Accuracy for bloom-560m ~ 35.36%
        print(f'Accuracy: {accuracy:0.4f}% ({num_corrects}/{num_requests}) ')
        return accuracy

class LLaMaLambadaEvaluator:
    def __init__(self, dataloader, tokenizer, device):
        self.data_loader = dataloader
        self.tokenizer = tokenizer
        self.device = device
        self.total_batches = -1

    def set_total_batches(self, total):
        self.total_batches = total

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        num_corrects = 0
        num_requests = 0
        count = 0
        for entries in tqdm(self.data_loader):
            input_token_ids, target_token_ids = entries['input_ids'], entries['target_ids']
            # input_token_ids are already padded
            batch_size = input_token_ids.shape[0]
            input_length = max([len(input) for input in input_token_ids])
            output_length = max([len(target) for target in target_token_ids])
            # Outputs (batch_size, seq_length)
            outputs = model.generate(inputs=input_token_ids.cuda(),
                                     max_new_tokens=output_length,
                                     do_sample=False,
                                     num_beams=1,
                                     temperature=1,
                                     top_k=1,
                                     top_p=0,
                                     repetition_penalty=1.0,
                                     length_penalty=1.0)

            # output_token_ids: input/padding/output
            output_token_ids = outputs[:, input_token_ids.shape[1]:]
            output_token_ids = [
                out[:len(tgt)].cpu()
                for out, tgt in zip(output_token_ids, target_token_ids)]

            # output_texts = self.tokenizer.batch_decode(output_token_ids)
            # target_texts = self.tokenizer.batch_decode(target_token_ids)
            # input_texts = self.tokenizer.batch_decode(input_token_ids)
            # Convert to output objects.
            for i in range(batch_size):
                out = output_token_ids[i]
                tgt = target_token_ids[i].cpu()
                if len(out) == len(tgt):
                    is_correct = (tgt == out).all()
                    num_corrects += int(is_correct)
            num_requests += batch_size
            count += 1
            if self.total_batches > 0 and count >= self.total_batches:
                break

        accuracy = num_corrects * 100 / num_requests
        # Reference: HF model's LAMBADA Accuracy for bloom-560m ~ 35.36%
        print(f'Accuracy: {accuracy:0.4f}% ({num_corrects}/{num_requests}) ')
        return accuracy