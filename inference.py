#!/usr/bin/env python3

import os
import argparse

# Configura o argumento --sotaque para escolher qual modelo regional testar
parser = argparse.ArgumentParser(description='Inferência do modelo G2P por sotaque/região.')
parser.add_argument('--sotaque', type=str, default=None, help='Nome do sotaque/região para inferência (ex: spx, rjx)')
parser.add_argument('--sentence', type=str, required=True, help='Frase para gerar a pronúncia')
parser.add_argument('--visualize', action='store_true', help='Salvar matriz de atenção')
args = parser.parse_args()

# Se um sotaque foi passado via terminal, atualiza a variável de ambiente antes de importar o config
if args.sotaque:
    os.environ['LANGUAGE'] = args.sotaque

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils.data import ParserLexicon
from model import Encoder, Decoder
from utils.config import DataConfig, ModelConfig, TestConfig
from utils.text_tools import tokenize_pt

# Carrega os dados e o vocabulário baseados no sotaque ativo
ds = ParserLexicon(
    DataConfig.graphemes_path,
    DataConfig.phonemes_path,
    DataConfig.lexicon_path
)

encoder_model = Encoder(
    ModelConfig.graphemes_size,
    ModelConfig.hidden_size
).to(TestConfig.device)
encoder_model.load_state_dict(torch.load(TestConfig.encoder_model_path, map_location=TestConfig.device))
encoder_model.eval()

decoder_model = Decoder(
    ModelConfig.phonemes_size,
    ModelConfig.hidden_size
).to(TestConfig.device)
decoder_model.load_state_dict(torch.load(TestConfig.decoder_model_path, map_location=TestConfig.device))
decoder_model.eval()

# Processamento da frase de entrada
words = tokenize_pt(args.sentence)
phonemes_result = []

for word in words:
    if word in ds.graphemes2idx:
        g = ds.graphemes2idx[word].unsqueeze(1).to(TestConfig.device)
        
        with torch.no_grad():
            enc = encoder_model(g)
            T, N = 1, 1
            hidden = torch.ones(1, N, ModelConfig.hidden_size).to(TestConfig.device)
            decoder_input = torch.tensor([[ds.sos_idx]], device=TestConfig.device)
            
            decoded_phonemes = []
            attentions = []
            
            for _ in range(100):  # Limite máximo de tamanho de saída
                out, hidden, attention = decoder_model(decoder_input, enc, hidden)
                attentions.append(attention.squeeze(0).cpu())
                
                pred_idx = out.argmax(dim=-1).item()
                if pred_idx == ds.eos_idx:
                    break
                
                decoded_phonemes.append(ds.idx2phonemes[pred_idx])
                decoder_input = torch.tensor([[pred_idx]], device=TestConfig.device)
                
            phonemes_result.append('|'.join(decoded_phonemes))
            
            # Visualização da atenção (separada por pasta de sotaque)
            if args.visualize and attentions:
                attentions = torch.stack(attentions).numpy()
                plt.figure(figsize=(10, 6))
                plt.imshow(attentions, aspect='auto', origin='lower')
                plt.xlabel('Encoder Timesteps (Graphemes)')
                plt.ylabel('Decoder Timesteps (Phonemes)')
                plt.title(f'Attention Weights - Sotaque: {DataConfig.language}')
                plt.colorbar()
                
                os.makedirs(f'attention/{DataConfig.language}', exist_ok=True)
                plt.savefig(f'attention/{DataConfig.language}/{word}.png')
                plt.close()
    else:
        phonemes_result.append(word)

print(' '.join(phonemes_result))