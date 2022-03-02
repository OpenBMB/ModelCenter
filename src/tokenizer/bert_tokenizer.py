import torch

from transformers import BertTokenizer as transformers_BertTokenizer

class BertTokenizer:
    def __init__(self):
        self.tokenizer = transformers_BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    def __call__(self, *args, **kwargs):
        logits = self.tokenizer(*args, **kwargs)
        batch_size, seq_len = logits['input_ids'].size()
        if batch_size % 2 != 0:
            raise Exception('batch size should be even')
        if seq_len % 2 != 0:
            logits['input_ids'] = torch.cat([logits['input_ids'], torch.zeros(batch_size, 1).to(torch.long)], dim=1)
            logits['token_type_ids'] = torch.cat([logits['token_type_ids'], torch.zeros(batch_size, 1).to(torch.long)], dim=1)
            logits['attention_mask'] = torch.cat([logits['attention_mask'], torch.zeros(batch_size, 1).to(torch.long)], dim=1)
        return logits