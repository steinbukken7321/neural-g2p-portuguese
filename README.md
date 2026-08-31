```markdown
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

### Inference

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

### Utilities

`batch_test.py` runs the trained model over a list of words (`list.txt` by default) and writes the results to a file, also scoped by `--sotaque`:

```
python batch_test.py --sotaque spx --list_path list.txt --output_path new_ipa.txt
```

`phone_batch.py` generates phonemes for a word list using `phonemizer`/`espeak` instead of the trained model — useful as an external reference baseline. Note that espeak-ng only distinguishes European (`pt`) from Brazilian (`pt-br`) Portuguese, so regions are mapped to the closest available variant:

```
python phone_batch.py --sotaque spx --list_path list.txt
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```