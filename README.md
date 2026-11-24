<div align="center">


# ANG2P

#### Accent-aware Neural Grapheme-to-Phoneme

</div>


🌐 **Language:** [🇺🇸 English](README.md) | [🇧🇷 Português](https://github.com/steinbukken7321/neural-g2p-portuguese/blob/main/README.pt-BR.md)

**ANG2P** (*Accent-aware Neural Grapheme-to-Phoneme*) is a neural Grapheme-to-Phoneme (G2P) project for Brazilian Portuguese focused on regional accent/dialect variation, interpretability through attention alignment, and practical usability for text-to-speech synthesis (TTS), automatic speech recognition (ASR), and Natural Language Processing (NLP).

## Objective

Grapheme-to-phoneme (G2P) conversion maps written words into phonetic transcriptions. Static pronunciation dictionaries often struggle with regional accents, lexical ambiguities, and out-of-vocabulary words. The objective of **ANG2P** is to provide an accessible, transparent, and extensible neural G2P model for Brazilian Portuguese variants, supporting accent-aware inference, phonetic representations (such as IPA and X-SAMPA), and evaluation workflows.

## Technologies and Project Structure

* **Core Stack:** Python, PyTorch (`requirements.txt`), NumPy/Matplotlib/tqdm, TensorBoard.


* **Model and Pipeline (`model.py`, `train.py`, `inference.py`):** Sequence-to-sequence neural G2P model with an attention mechanism (`attention/attention-bidi.jpg`), training pipeline, and inference script with optional X-SAMPA output and attention visualization (`--visualize`).


* **Batch and Accent Testing (`batch_test.py`, `test_all_accents.py`, `list.txt`):** Scripts for batch inference (`results/resultado_<accent>.txt`, with `spx` as default accent) and testing across different accents/variants.


* **Data, Lexicons and Evaluation (`CSVs/`, `resources/`, `utils/`, `evaluation/`):** CSV/lexicon processing, regional phonetic resources, X-SAMPA conversion utilities, and evaluation metrics/tests.



## Dataset

The dataset was collected from [Portal da Língua Portuguesa](http://www.portaldalinguaportuguesa.org/), complemented with custom entries to broaden coverage of everyday Brazilian Portuguese words. Some ambiguities were also resolved to reflect a specific speaker bias, with the dictionary for São Paulo speakers chosen as a baseline.

As in [hajix/G2P](https://github.com/hajix/G2P), on which this implementation was based, you can easily provide and use your own language-specific pronunciation dictionary for training G2P. More details about data preparation and contributions can be found in `resources/` and `CSVs/`.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/steinbukken7321/neural-g2p-portuguese.git
cd neural-g2p-portuguese
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
```

3. Install the dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage and Commands

### Training and Monitoring

To start training the neural G2P model using the project configuration and dataset/lexicon resources:

```bash
python train.py
```

To launch TensorBoard and monitor training and validation curves stored in the `log` directory:

```bash
tensorboard --logdir log --bind_all
```

Training parameters can be found in `config.py`.

### Inference and Attention Visualization

To run G2P inference on an input sentence and get the segmented phonetic transcription:

```bash
# PT-BR example
python inference.py --sentence 'olá, vamos testar esse projeto.'
o|l|a| |,| |v|a|m|ʊ|s| |t|e|s|t|a| |e|s|i| |p|ɾ|o|ʒ|e|t|ʊ| |.
```

To run inference while generating/displaying attention weight visualizations (`--visualize`) and inspect how graphemes align with generated phonemes:

```bash
# PT-BR example
python inference.py --visualize --sentence 'olá, vamos testar esse projeto.'
o|l|a| |,| |v|a|m|ʊ|s| |t|e|s|t|a| |e|s|i| |p|ɾ|o|ʒ|e|t|ʊ| |.
```

To check optional parameters implemented in the script, such as output format (IPA / X-SAMPA) or accent selection (`--sotaque`):

```bash
python inference.py --help
```

### Batch and Accent Testing

To run batch G2P tests over input lists (such as `list.txt`), saving results to `results/resultado_<accent>.txt` (using `spx` as default accent):

```bash
python batch_test.py
```

To run tests across all available accents/variants, streaming output directly to the terminal:

```bash
python test_all_accents.py
```

## Contributing

Contributions are welcome from researchers, linguists, and developers interested in Brazilian Portuguese G2P, regional accents, and phonetic resources. You can contribute by:

1. Forking the repository on GitHub.


2. Creating a feature or fix branch (`git checkout -b feature/your-improvement`).
3. Adding or improving pronunciation dictionaries, regional variants, evaluation scripts, or documentation (`resources/`, `CSVs/`, `evaluation/`).


4. Opening an **Issue** to report bugs, suggest features, or discuss data/accent improvements.


5. Submitting a **Pull Request** with a clear description of your changes and how they were tested.



## Frequently Asked Questions (FAQ)

* **What does ANG2P stand for?**
**ANG2P** stands for *Accent-aware Neural Grapheme-to-Phoneme*, a neural G2P model for Brazilian Portuguese designed with attention to regional variation and interpretability.


* **Why support both IPA and X-SAMPA?**
IPA is the standard phonetic alphabet for acoustic and articulatory description, while ASCII-based mappings such as X-SAMPA help avoid encoding/Unicode issues in command-line tools and legacy speech/NLP pipelines.


* **Can I train ANG2P with my own pronunciation dictionary or regional data?**
Yes. You can provide custom pronunciation dictionaries and resources following the structure in `resources/` and `CSVs/`, adjusting training settings before running `python train.py`.


* **How can I inspect how the model aligns graphemes and phonemes?**
Run `python inference.py --visualize --sentence "..."` to generate attention visualizations and inspect model alignment behavior.



## References and Credits

This project builds upon and references prior work and repositories:

* [hajix/G2P](https://github.com/hajix/G2P) — Original attention-based neural G2P reference implementation adapted for this work.


* [fabianoluzbr/neural-g2p-portuguese](https://github.com/fabianoluzbr/neural-g2p-portuguese) — Prior Portuguese neural G2P work and resources.



## Contact

For questions, research collaborations, or issues related to this repository:

| Role | Name / Channel |
| --- | --- |
| **Repository** | [steinbukken7321/neural-g2p-portuguese](https://github.com/steinbukken7321/neural-g2p-portuguese)|
| **Supervisor** | *(add name here)* |
| **Author** | **Rafael Ziani de Carvalho** (`@steinbukken7321`)|

## License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.