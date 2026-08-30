import os
import json

import torch

cpu = torch.device('cpu')
gpu = torch.device('cuda')


class DataConfig(object):
    language = os.getenv('LANGUAGE', 'spx')
    
    # Caminhos globais compartilhados
    graphemes_path = 'resources/global/Graphemes.json'
    phonemes_path = 'resources/global/Phonemes.json'
    
    # Caminho dinâmico para o léxico da região ativa
    lexicon_path = f'resources/lexicons/{language}.json'


class ModelConfig(object):
    with open(DataConfig.graphemes_path, encoding='utf-8') as f:
        graphemes_size = len(json.load(f))

    with open(DataConfig.phonemes_path, encoding='utf-8') as f:
        phonemes_size = len(json.load(f))

    hidden_size = 256


class TrainConfig(object):
    device = gpu if torch.cuda.is_available() else cpu
    lr = 2e-4
    batch_size = 64
    epochs = int(os.getenv('EPOCHS', '30'))
    # Log organizado por sotaque/região (ex: log/spx)
    log_path = f'log/{DataConfig.language}'


class TestConfig(object):
    device = cpu
    # Checkpoints organizados por sotaque/região (ex: checkpoints/spx/encoder_e30.pth)
    encoder_model_path = f'checkpoints/{DataConfig.language}/encoder_e{TrainConfig.epochs:02}.pth'
    decoder_model_path = f'checkpoints/{DataConfig.language}/decoder_e{TrainConfig.epochs:02}.pth'