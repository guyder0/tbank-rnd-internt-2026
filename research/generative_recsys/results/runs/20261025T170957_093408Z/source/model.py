"""Small time-aware SID encoder-decoder shared by CPU and CUDA."""
import math

import numpy as np
import torch
from torch import nn

from research.generative_recsys.src.data import cap_history
from research.generative_recsys.src.tokenizer import BOS, PAD, UNK


class SIDGenerator(nn.Module):
    def __init__(self, vocab_size, sid_length, config):
        super().__init__()
        dimension = config['d_model']
        self.sid_length = sid_length
        self.embedding = nn.Embedding(vocab_size, dimension, padding_idx=PAD)
        self.time = nn.Sequential(nn.Linear(1, dimension), nn.Tanh())
        self.summary = nn.Parameter(torch.randn(dimension)*.02)
        self.position = nn.Embedding(sid_length, dimension)
        encoder = nn.TransformerEncoderLayer(dimension, config['heads'], dimension*4,
                    dropout=config['dropout'], batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder, config['layers'], enable_nested_tensor=False)
        decoder = nn.TransformerDecoderLayer(dimension, config['heads'], dimension*4,
                    dropout=config['dropout'], batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder, config['layers'])
        self.output_norm = nn.LayerNorm(dimension)
        self.output = nn.Linear(dimension, vocab_size)

    def encode(self, history, ages, padding):
        # Code levels have disjoint token ranges; whole objects are unordered within a date.
        x = self.embedding(history).sum(2)/math.sqrt(self.sid_length)
        x = x+self.time(torch.log1p(ages).unsqueeze(-1)/math.log(366))
        x = torch.cat([self.summary[None, None].expand(len(x), 1, -1), x], dim=1)
        padding = torch.cat([torch.zeros((len(x), 1), dtype=torch.bool, device=x.device), padding], dim=1)
        return self.encoder(x, src_key_padding_mask=padding), padding

    def decode(self, memory, memory_padding, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embedding(tokens)+self.position(positions)[None]
        mask = torch.triu(torch.ones((len(positions), len(positions)), dtype=torch.bool,
                                    device=tokens.device), diagonal=1)
        hidden = self.decoder(x, memory, tgt_mask=mask, memory_key_padding_mask=memory_padding)
        return self.output(self.output_norm(hidden))

    def forward(self, history, ages, padding, targets):
        memory, memory_padding = self.encode(history, ages, padding)
        inputs = torch.cat([torch.full_like(targets[:, :1], BOS), targets[:, :-1]], dim=1)
        return self.decode(memory, memory_padding, inputs)


def history_batch(queries, mapping, length, limit, device):
    histories = [cap_history(query, limit) for query in queries]
    size = max(1, max(len(items) for items, _ in histories))
    codes = np.zeros((len(queries), size, length), dtype='int64')
    ages = np.zeros((len(queries), size), dtype='float32')
    padding = np.ones((len(queries), size), dtype=bool)
    for row, (items, times) in enumerate(histories):
        for col, (item, age) in enumerate(zip(items, times)):
            if age < 1:
                raise ValueError('History is not strictly earlier than cutoff')
            codes[row, col] = mapping.get(int(item), np.full(length, UNK))
            ages[row, col] = age
            padding[row, col] = False
    return (torch.tensor(codes, device=device), torch.tensor(ages, device=device),
            torch.tensor(padding, device=device))


def decoder_callback(model, query, mapping, config, device):
    tensors = history_batch([query], mapping, model.sid_length, config['max_history'], device)
    memory, padding = model.encode(*tensors)

    def probabilities(prefixes):
        tokens = torch.tensor([[BOS, *prefix] for prefix in prefixes], dtype=torch.long, device=device)
        logits = model.decode(memory.expand(len(prefixes), -1, -1),
                              padding.expand(len(prefixes), -1), tokens)[:, -1]
        return logits.log_softmax(-1).detach().cpu().numpy()
    return probabilities
