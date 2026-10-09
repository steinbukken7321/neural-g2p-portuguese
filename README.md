# Neural G2P to portuguese language

Grapheme-to-phoneme (G2P) conversion is the process of generating pronunciation for words based on their written form. It has a highly essential role for natural language processing, text-to-speech synthesis and automatic speech recognition systems. This project was adapted from [https://github.com/hajix/G2P](https://github.com/hajix/G2P).

## Credits

This project builds on two prior works:
Originates from [hajix/G2P](https://github.com/hajix/G2P) and [fabianoluzbr/neural-g2p-portuguese](https://github.com/fabianoluzbr/neural-g2p-portuguese)

## Dependencies

The following libraries are used:
- pytorch
- tqdm
- matplotlib
- tensorboard
- phonemizer (requires [espeak-ng](https://github.com/espeak-ng/espeak-ng) installed on the system — not installable via pip)
- packaging

Install Python dependencies using pip:

```
pip3 install -r requirements.txt
```

## Dataset & Regional Variations

The dataset used here was taken from site [http://www.portaldalinguaportuguesa.org/](http://www.portaldalinguaportuguesa.org/), as well as some insertions made by me so that the dataset would give more coverage to common words in the daily life of the Brazilian Portuguese. Some ambiguities were also resolved as the intent of this dataset is to contain a specific speaker bias.

The project supports **multiple regional accents and dialects** via the `--sotaque` flag. Graphemes and phonemes are shared across all accents (`resources/global/Graphemes.json`, `resources/global/Phonemes.json`), while each accent has its own lexicon under `resources/lexicons/` (e.g., `spx.json`, `rjx.json`), allowing you to train, evaluate, and test models tailored to specific regional variants independently. More details about data preparation and contribution could be found in `resources`.

Supported regions:

| Region | Code |
|---|---|
| Luanda | lda |
| Rio de Janeiro (non-standard) | rjo |
| Rio de Janeiro (standard) | rjx |
| São Paulo (standard) | spx |
| São Paulo (non-standard) | spo |
| Maputo (non-standard) | map |
| Maputo (standard) | mpx |
| Lisbon (standard) | lbx |
| Lisbon (non-standard) | lbn |
| Dili | dli |

## Attention Model

Both encoder-decoder seq2seq model and attention model could handle G2P problem.
Here we train attention based model.

The encoder model get sequence of graphemes and produces states at each timestep.
Encoder states used during attention decoding.
The decoder attends to appropriate encoder state (according to its state) and produces phonemes.

### Train

To start training the model for a specific accent or region, use the `--sotaque` flag:

```
python train.py --sotaque spx
```

*(If omitted, it defaults to the configuration defined in `utils/config.py`).*

Checkpoints are automatically organized per region under `checkpoints/<sotaque>/`, and logs are saved under `log/<sotaque>/`. You can use tensorboard to check the training loss for a specific region:

```
tensorboard --logdir log --bind_all
```

Training parameters could be found at `utils/config.py`.

## Inference

To get the pronunciation of a sentence using a specific trained regional model, pass the `--sotaque` flag during inference:

```
# Example testing the spx accent
python inference.py --sotaque spx --sentence 'olá, vamos testar esse projeto.'
o|l|a| |,| |v|a|m|ʊ|s| |t|e|s|t|a| |e|s|i| |p|ɾ|o|ʒ|e|t|ʊ| |.
```

You could also visualize the attention weights using `--visualize`, which saves the plot inside an attention directory structured by the selected region (`attention/<sotaque>/<word>.png`):

```
# Example with visualization for spx accent
python inference.py --sotaque spx --visualize --sentence 'olá, vamos testar esse projeto.'
o|l|a| |,| |v|a|m|ʊ|s| |t|e|s|t|a| |e|s|i| |p|ɾ|o|ʒ|e|t|ʊ| |.
```

#### Output format: IPA or X-SAMPA

The `--format` flag selects the notation of the output: `ipa` (default) or `xsampa`.

```
# IPA (default, same as omitting --format)
python inference.py --sotaque spx --format ipa --sentence 'olá, vamos testar esse projeto.'
o|l|a| |,| |v|a|m|ʊ|s| |t|e|s|t|a| |e|s|i| |p|ɾ|o|ʒ|e|t|ʊ| |.

# X-SAMPA
python inference.py --sotaque spx --format xsampa --sentence 'olá, vamos testar esse projeto.'
o|l|a| |,| |v|a|m|U|s| |t|e|s|t|a| |e|s|i| |p|4|o|Z|e|t|U| |.
```

The model always predicts IPA; X-SAMPA is produced from it by the one-to-one mapping in `utils/phonetic_maps.py`, so both outputs have the same number of tokens. Spaces and punctuation are kept as they are.

Sim, é necessário fazer pequenas atualizações para deixar a documentação 100% alinhada com as últimas mudanças do projeto:

1. **Nome padrão do arquivo de saída em `Batch Testing & Utilities**`: Atualizar de `results/transcricoes_ipa.txt` para `results/resultado_<sotaque>.txt` (ex: `results/resultado_spx.txt`), informando que o sotaque padrão agora é `spx`.
2. **Remoção de referências legadas em `Tests debug**`: Remover a menção a `phone_batch.py` (já deletado) e atualizar `test_all.py` para `test_all_accents.py` (conforme exibido na árvore de arquivos da sua IDE).

---

### Batch Testing & Utilities

`batch_test.py` executes the trained model over a list of words (`list.txt` by default) for a specified regional accent (`--sotaque`, defaulting to `spx`). Output files are automatically saved into the `/results` directory with the accent code embedded in the filename:

```bash
# Basic batch test (defaults to --sotaque spx, outputs to results/resultado_spx.txt)
python batch_test.py

# Batch test specifying accent, custom output filename, and attention map generation
python batch_test.py --sotaque rjx --list_path list.txt --output_path meu_resultado.txt --visualize

```

**Available Options:**

* `--sotaque`: Regional accent code to evaluate (default: `spx`).
* `--list_path`: Path to the input word list (default: `list.txt`).
* `--output_path`: Filename for predictions saved in `/results` (default: `resultado_<sotaque>.txt`).
* `--visualize`: Saves attention heatmaps for each word in `/attention/<sotaque>/`.

---
### Tests debug

`test_all_accents.py` is a utility script that runs evaluation checks across regional models, testing both **IPA and X-SAMPA** outputs (plus `--visualize` and `batch_test.py`) and verifying the phonetic mappings:

```bash
# Test all available accent checkpoints
python test_all_accents.py

# Test specific regions
python test_all_accents.py --regions spx rjx

```

For each region, it verifies that:

* `inference.py --format ipa` retains spaces and punctuation;
* `inference.py --format xsampa` produces a valid token-by-token mapped output;
* `inference.py --visualize` generates attention heatmaps under `attention/<sotaque>/`;
* `batch_test.py` creates structured predictions under `/results`.




---
### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.