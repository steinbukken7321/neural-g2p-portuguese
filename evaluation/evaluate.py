#!/usr/bin/env python3
import os
import argparse
import json
import random

parser = argparse.ArgumentParser(description='Avaliação de acurácia do modelo G2P por sotaque/região.')
parser.add_argument('--sotaque', type=str, default=None,
                     help='Código do sotaque/região a avaliar (ex: spx, rjx)')
parser.add_argument('--sample_size', type=int, default=None,
                     help='Avalia apenas uma amostra aleatória do léxico (padrão: todas as palavras)')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--report_path', type=str, default=None,
                     help='Caminho do CSV com as divergências (padrão: eval_<sotaque>.csv)')
args = parser.parse_args()

if args.sotaque:
    os.environ['SOTAQUE'] = args.sotaque

import torch
from model import Encoder, Decoder
from utils.data import ParserLexicon
from utils.config import DataConfig, ModelConfig, TestConfig


def load_model(model_path, model):
    model.load_state_dict(torch.load(
        model_path, map_location=lambda storage, loc: storage
    ))
    model.to(TestConfig.device)
    model.eval()
    return model


class G2P(object):
    def __init__(self):
        self.ds = ParserLexicon(
            DataConfig.graphemes_path,
            DataConfig.phonemes_path,
            DataConfig.lexicon_path
        )
        self.encoder_model = Encoder(ModelConfig.graphemes_size, ModelConfig.hidden_size)
        load_model(TestConfig.encoder_model_path, self.encoder_model)
        self.decoder_model = Decoder(ModelConfig.phonemes_size, ModelConfig.hidden_size)
        load_model(TestConfig.decoder_model_path, self.decoder_model)

    def __call__(self, word):
        x = [0] + [self.ds.g2idx[ch] for ch in word] + [1]
        x = torch.tensor(x).long().unsqueeze(1)
        with torch.no_grad():
            enc = self.encoder_model(x)
        phonemes = []
        x = torch.zeros(1, 1).long().to(TestConfig.device)
        hidden = torch.ones(1, 1, ModelConfig.hidden_size).to(TestConfig.device)
        while True:
            with torch.no_grad():
                out, hidden, _ = self.decoder_model(x, enc, hidden)
            max_index = out[0, 0].argmax()
            x = max_index.unsqueeze(0).unsqueeze(0)
            phonemes.append(self.ds.idx2p[max_index.item()])
            if max_index.item() == 1:
                break
        return phonemes[:-1]  # remove <eos>


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def main():
    with open(DataConfig.lexicon_path, encoding='utf-8') as f:
        lexicon = json.load(f)

    if args.sample_size and args.sample_size < len(lexicon):
        random.seed(args.seed)
        lexicon = random.sample(lexicon, args.sample_size)

    g2p = G2P()

    total = exact_matches = total_phonemes = total_edits = 0
    mismatches = []

    for word, expected in lexicon:
        expected_phonemes = [p for p in expected.split(' ') if p != '']
        predicted = g2p(word)

        total += 1
        total_phonemes += len(expected_phonemes)
        edits = levenshtein(predicted, expected_phonemes)
        total_edits += edits

        if predicted == expected_phonemes:
            exact_matches += 1
        else:
            mismatches.append((word, ' '.join(expected_phonemes), ' '.join(predicted)))

    word_accuracy = 100 * exact_matches / total if total else 0
    phoneme_error_rate = 100 * total_edits / total_phonemes if total_phonemes else 0

    print(f'Sotaque: {DataConfig.sotaque}')
    print(f'Palavras avaliadas: {total}')
    print(f'Word Accuracy (match exato): {word_accuracy:.2f}%')
    print(f'Phoneme Error Rate: {phoneme_error_rate:.2f}%')

    report_path = args.report_path or f'eval_{DataConfig.sotaque}.csv'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('palavra,esperado,previsto\n')
        for word, expected, predicted in mismatches:
            f.write(f'{word},{expected},{predicted}\n')

    print(f'{len(mismatches)} divergências salvas em {report_path}')


if __name__ == '__main__':
    main()