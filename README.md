# Neural G2P to portuguese language

Grapheme-to-phoneme (G2P) conversion is the process of generating pronunciation for words based on their written form. It has a highly essential role for natural language processing, text-to-speech synthesis and automatic speech recognition systems. This project was adapted from [https://github.com/hajix/G2P](https://github.com/hajix/G2P).

## Credits

This project builds on two prior works:
Originates from [hajix/G2P](https://github.com/hajix/G2P) and [fabianoluzbr/neural-g2p-portuguese](https://github.com/fabianoluzbr/neural-g2p-portuguese)

## Dependencies

The following libraries are used:
pytorch
tqdm
matplotlib

Install dependencies using pip:

```
pip3 install -r requirements.txt

```

## Dataset & Regional Variations

The dataset used here was taken from site [http://www.portaldalinguaportuguesa.org/](http://www.portaldalinguaportuguesa.org/), as well as some insertions made by me so that the dataset would give more coverage to common words in the daily life of the Brazilian Portuguese. Some ambiguities were also resolved as the intent of this dataset is to contain a specific speaker bias.

The project now supports **multiple regional accents and dialects**. The lexicons are structured internally inside `resources/lexicons/` (e.g., `spx.json`, `rjx.json`), allowing you to train, evaluate, and test models tailored to specific regional variants independently. More details about data preparation and contribution could be found in `resources`.

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

You could also visualize the attention weights using `--visualize`, which saves the plot inside an attention directory structured by the selected language/region (`attention/<sotaque>/<word>.png`):

```
# Example with visualization for spx accent
python inference.py --sotaque spx --visualize --sentence 'olá, vamos testar esse projeto.'
o|l|a| |,| |v|a|m|ʊ|s| |t|e|s|t|a| |e|s|i| |p|ɾ|o|ʒ|e|t|ʊ| |.

```

## License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.