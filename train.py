#!/usr/bin/env python3
import os
import argparse

parser = argparse.ArgumentParser(description='Treinamento do modelo G2P por sotaque/região.')
parser.add_argument('--sotaque', type=str, default=None,
                     help='Código do sotaque/região a treinar (ex: spx, rjx, lbx...)')
args = parser.parse_args()

if args.sotaque:
    os.environ['SOTAQUE'] = args.sotaque

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.data import ParserLexicon, collate_fn
from model import Encoder, Decoder
from utils.config import DataConfig, ModelConfig, TrainConfig

os.makedirs(TrainConfig.checkpoints_path, exist_ok=True)
os.makedirs(TrainConfig.log_path, exist_ok=True)

# data prep
ds = ParserLexicon(
        DataConfig.graphemes_path,
        DataConfig.phonemes_path,
        DataConfig.lexicon_path
    )
dl = DataLoader(ds, collate_fn=collate_fn, batch_size=TrainConfig.batch_size)

# models
encoder_model = Encoder(ModelConfig.graphemes_size, ModelConfig.hidden_size).to(TrainConfig.device)
decoder_model = Decoder(ModelConfig.phonemes_size, ModelConfig.hidden_size).to(TrainConfig.device)

log = SummaryWriter(TrainConfig.log_path)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    list(encoder_model.parameters()) + list(decoder_model.parameters()),
    lr=TrainConfig.lr
)

counter = 0
for e in range(TrainConfig.epochs):
    print('-' * 20 + f'epoch: {e+1:02d}' + '-' * 20)
    for g, p in tqdm(dl):
        g = g.to(TrainConfig.device)
        p = p.to(TrainConfig.device)
        enc = encoder_model(g)
        T, N = p.size()
        outputs = []
        hidden = torch.ones(1, N, ModelConfig.hidden_size).to(TrainConfig.device)
        for t in range(T - 1):
            out, hidden, _ = decoder_model(p[t:t+1], enc, hidden)
            outputs.append(out)
        outputs = torch.cat(outputs)
        outputs = outputs.view((T-1) * N, -1)
        p = p[1:].view(-1)
        loss = criterion(outputs, p)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log.add_scalar('loss', loss.item(), counter)
        counter += 1
    torch.save(encoder_model.state_dict(), f'{TrainConfig.checkpoints_path}/encoder_e{e+1:02d}.pth')
    torch.save(decoder_model.state_dict(), f'{TrainConfig.checkpoints_path}/decoder_e{e+1:02d}.pth')