import os
import json
import torch

cpu = torch.device('cpu')
gpu = torch.device('cuda')


class DataConfig(object):
    # Sotaque/região ativa (ex: spx, rjx, lbx...). Precisa ser definida via
    # --sotaque ANTES deste módulo ser importado (train.py e inference.py
    # já fazem isso, setando os.environ['SOTAQUE'] antes do import).
    sotaque = os.getenv('SOTAQUE', 'spx')

    # Grafemas e fonemas são compartilhados entre todos os sotaques
    graphemes_path = 'resources/global/Graphemes.json'
    phonemes_path = 'resources/global/Phonemes.json'

    # Léxico é específico de cada sotaque
    lexicon_path = f'resources/lexicons/{sotaque}.json'


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
    log_path = f'log/{DataConfig.sotaque}'
    checkpoints_path = f'checkpoints/{DataConfig.sotaque}'


class TestConfig(object):
    device = cpu
    encoder_model_path = f'checkpoints/{DataConfig.sotaque}/encoder_e{TrainConfig.epochs:02}.pth'
    decoder_model_path = f'checkpoints/{DataConfig.sotaque}/decoder_e{TrainConfig.epochs:02}.pth'