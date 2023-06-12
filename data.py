import torch
import transformers
import json

class LambadaDataset(torch.utils.data.Dataset):
    """ LAMBADA dataset class. """

    def __init__(self,
                 path: str,
                 tokenizer: transformers.PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        with open(path, 'r') as f:
            inputs, targets = zip(*[json.loads(line)["text"] .strip('\n').rsplit(' ', 1) for line in f.readlines()])
            # This whitespace preprocessing (additional space to the target)
            # is required.
            targets = [' ' + tgt for tgt in targets]
            self.encodings = self.tokenizer(list(inputs),
                                            targets,
                                            padding=True,
                                            return_token_type_ids=True,
                                            return_tensors='pt')

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return dict(
            input_ids=self.encodings['input_ids'][idx],
            attention_mask=self.encodings['attention_mask'][idx],
            token_type_ids=self.encodings['token_type_ids'][idx]
        )

class GLMLambadaDataset(torch.utils.data.Dataset):
    """ LAMBADA dataset class. """

    def __init__(self,
                 path: str,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 split=-1):
        self.tokenizer = tokenizer
        self.encodings = dict()
        self.encodings['input_ids'] = []
        self.encodings['target_ids'] = []
        with open(path, 'r') as f:
            inputs, targets = zip(*[json.loads(line)["text"] .strip('\n').rsplit(' ', 1) for line in f.readlines()][:split])
            for input, target in zip(inputs, targets):
                input_token_ids = self.tokenizer(input, padding=True, return_token_type_ids=False, return_tensors='pt')
                self.encodings['input_ids'].append(input_token_ids['input_ids'])
                target_token_ids = self.tokenizer(target, padding=True, return_token_type_ids=False, return_tensors='pt')
                # remove suffix
                self.encodings['target_ids'].append(target_token_ids['input_ids'][:,:-2])

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return dict(
            input_ids=self.encodings['input_ids'][idx],
            target_ids=self.encodings['target_ids'][idx]
        )

class LLaMaLambadaDataset(torch.utils.data.Dataset):
    """ LAMBADA dataset class. """

    def __init__(self,
                 path: str,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 split=5153):
        self.tokenizer = tokenizer
        self.encodings = dict()
        self.encodings['input_ids'] = []
        self.encodings['target_ids'] = []
        with open(path, 'r') as f:
            inputs, targets = zip(*[json.loads(line)["text"] .strip('\n').rsplit(' ', 1) for line in f.readlines()][:split])
            for input, target in zip(inputs, targets):
                input_token_ids = self.tokenizer(input, padding=True, return_token_type_ids=False, return_tensors='pt')
                # input_token_ids = self.tokenizer(input, return_tensors='pt')
                # input_token_ids = self.tokenizer.encode(input, return_tensors="pt")
                self.encodings['input_ids'].append(input_token_ids['input_ids'])
                target_token_ids = self.tokenizer(target, padding=True, return_token_type_ids=False, return_tensors='pt')
                # target_token_ids = self.tokenizer(target, return_tensors='pt')
                # target_token_ids = self.tokenizer.encode(target, return_tensors="pt")
                # remove suffix
                self.encodings['target_ids'].append(target_token_ids['input_ids'])

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return dict(
            input_ids=self.encodings['input_ids'][idx],
            target_ids=self.encodings['target_ids'][idx]
        )

class OPTLambadaDataset(torch.utils.data.Dataset):
    """ LAMBADA dataset class. """

    def __init__(self,
                 path: str,
                 tokenizer: transformers.PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.encodings = dict()
        self.encodings['input_ids'] = []
        self.encodings['target_ids'] = []
        with open(path, 'r') as f:
            inputs, targets = zip(*[json.loads(line)["text"] .strip('\n').rsplit(' ', 1) for line in f.readlines()])
            # add space for right generation
            targets = [' ' + target for target in targets]
            for input, target in zip(inputs, targets):
                input_token_ids = self.tokenizer(input, padding=True, return_token_type_ids=False, return_tensors='pt')
                self.encodings['input_ids'].append(input_token_ids['input_ids'])
                target_token_ids = self.tokenizer(target, padding=True, return_token_type_ids=False, return_tensors='pt')
                # remove prefix
                self.encodings['target_ids'].append(target_token_ids['input_ids'][:, 1:])

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return dict(
            input_ids=self.encodings['input_ids'][idx],
            target_ids=self.encodings['target_ids'][idx]
        )
