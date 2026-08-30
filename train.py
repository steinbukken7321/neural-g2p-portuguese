#!/usr/bin/env python3

import os
import argparse

# Configura o argumento --sotaque para escolher a variação regional pela linha de comando
parser = argparse.ArgumentParser(description='Treinamento do modelo G2P por sotaque/região.')
parser.add_argument('--sotaque', type=str, default=None, help='Nome do sotaque/região (ex: spx, rjx)')
args = parser.parse_args()

# Se um sotaque foi passado via terminal, atualiza a variável de ambiente antes de importar o config
if args.sotaque:
    os.environ['LANGUAGE'] = args.sotaque

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.data import ParserLexicon, collate_fn
from model import Encoder, Decoder
from utils.config import DataConfig, ModelConfig, TrainConfig

# to save results
os.makedirs('checkpoints', exist_ok=True)

# data prep
ds = ParserLexicon(
    DataConfig.graphemes_path,
    DataConfig.phonemes_path,
    DataConfig.lexicon_path
)
dl = DataLoader(
    ds,
    collate_fn=collate_fn,
    batch_size=TrainConfig.batch_size
)

# models
encoder_model = Encoder(
    ModelConfig.graphemes_size,
    ModelConfig.hidden_size
).to(TrainConfig.device)
decoder_model = Decoder(
    ModelConfig.phonemes_size,
    ModelConfig.hidden_size
).to(TrainConfig.device)

# log
log = SummaryWriter(TrainConfig.log_path)

# loss
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(
    list(encoder_model.parameters()) +
    list(decoder_model.parameters()),
    lr=TrainConfig.lr
)

# Garante que a pasta de checkpoints da região específica exista
checkpoint_dir = f'checkpoints/{DataConfig.language}'
os.makedirs(checkpoint_dir, exist_ok=True)

# training loop
counter = 0
for e in range(TrainConfig.epochs):
    print('-' * 20 + f'epoch: {e+1:02d}' + '-' * 20)
    for g, p in tqdm(dl):
        g = g.to(TrainConfig.device)
        p = p.to(TrainConfig.device)
        # encode
        enc = encoder_model(g)

        # decoder
        T, N = p.size()
        outputs = []
        hidden = torch.ones(
            1,
            N,
            ModelConfig.hidden_size
        ).to(TrainConfig.device)
        for t in range(T - 1):
            out, hidden, _ = decoder_model(
                p[t:t+1],
                enc,
                hidden
            )
            outputs.append(out)
        outputs = torch.cat(outputs)

        # flat Time and Batch, calculate loss
        outputs = outputs.view((T-1) * N, -1)
        p = p[1:]  # trim first phoneme
        p = p.view(-1)
        loss = criterion(outputs, p)

        # update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log.add_scalar('loss', loss.item(), counter)
        counter += 1

    # save model separated by region folder
    torch.save(
        encoder_model.state_dict(),
        f'{checkpoint_dir}/encoder_e{e+1:02d}.pth'
    )
    torch.save(
        decoder_model.state_dict(),
        f'{checkpoint_dir}/decoder_e{e+1:02d}.pth'
    )