<div align="center">

# ANG2P

#### Conversão Neural Grafema-Fonema Sensível a Sotaques


</div>

🌐 **Idioma:** [🇧🇷 Português](https://www.google.com/search?q=README.pt-BR.md) | [🇺🇸 English](README.md)

**ANG2P** (*Accent-aware Neural Grapheme-to-Phoneme*) é um projeto de conversão neural Grafema-Fonema (G2P) para o português brasileiro focado em variações regionais de sotaque/dialeto, interpretabilidade por meio do alinhamento de atenção e usabilidade prática para síntese de fala (TTS), reconhecimento automático de fala (ASR) e Processamento de Linguagem Natural (PLN).

## Objetivo

A conversão grafema-fonema (G2P) mapeia palavras escritas para suas respectivas transcrições fonéticas. Dicionários de pronúncia estáticos costumam ter dificuldades com sotaques regionais, ambiguidades lexicais e palavras fora do vocabulário. O objetivo do **ANG2P** é fornecer um modelo G2P neural acessível, transparente e extensível para variantes do português brasileiro, oferecendo suporte a inferência sensível a sotaques, representações fonéticas (como IPA e X-SAMPA) e fluxos de avaliação.

## Tecnologias e Estrutura do Projeto

* **Stack Principal:** Python, PyTorch (`requirements.txt`), NumPy/Matplotlib/tqdm, TensorBoard.


* **Modelo e Pipeline (`model.py`, `train.py`, `inference.py`):** Modelo G2P neural sequência-a-sequência com mecanismo de atenção (`attention/attention-bidi.jpg`), pipeline de treinamento e script de inferência com saída opcional em X-SAMPA e visualização de atenção (`--visualize`).


* **Testes em Lote e por Sotaque (`batch_test.py`, `test_all_accents.py`, `list.txt`):** Scripts para inferência em lote (`results/resultado_<sotaque>.txt`, com sotaque padrão `spx`) e testes entre diferentes sotaques/variantes.


* **Dados, Léxicos e Avaliação (`CSVs/`, `resources/`, `utils/`, `evaluation/`):** Processamento de CSVs/léxicos, recursos fonéticos regionais, utilitários de conversão X-SAMPA e métricas/testes de avaliação.



## Conjunto de Dados (Dataset)

O conjunto de dados utilizado foi obtido a partir do [Portal da Língua Portuguesa](http://www.portaldalinguaportuguesa.org/), complementado por inserções próprias para ampliar a cobertura de palavras comuns do cotidiano do português brasileiro. Algumas ambiguidades também foram resolvidas com o objetivo de refletir um viés específico de falante, tendo sido escolhido como base o dicionário de falantes de São Paulo.

Assim como no [hajix/G2P](https://github.com/hajix/G2P), no qual esta implementação foi baseada, você pode facilmente fornecer e utilizar seu próprio dicionário de pronúncia específico para treinar o G2P. Mais detalhes sobre preparação de dados e contribuição podem ser encontrados em `resources/` e `CSVs/`.

## Instalação

1. Clone o repositório:

```bash
git clone https://github.com/steinbukken7321/neural-g2p-portuguese.git
cd neural-g2p-portuguese
```

2. Crie e ative um ambiente virtual (recomendado):

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
```

3. Instale as dependências:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Uso e Comandos

### Treinamento e Monitoramento

Para iniciar o treinamento do modelo neural G2P utilizando as configurações do projeto e os recursos de dataset/léxico:

```bash
python train.py
```

Para iniciar o TensorBoard e acompanhar as curvas de treinamento e validação armazenadas no diretório `log`:

```bash
tensorboard --logdir log --bind_all
```

Os parâmetros de treinamento podem ser encontrados em `config.py`.

### Inferência e Visualização de Atenção

Para executar a inferência G2P em uma frase de entrada e obter a transcrição fonética segmentada:

```bash
# Exemplo em PT-BR
python inference.py --sentence 'olá, vamos testar esse projeto.'
o|l|a| |,| |v|a|m|ʊ|s| |t|e|s|t|a| |e|s|i| |p|ɾ|o|ʒ|e|t|ʊ| |.
```

Para executar a inferência enquanto gera/exibe visualizações dos pesos de atenção (`--visualize`) e inspecionar como os grafemas se alinham aos fonemas gerados:

```bash
# Exemplo em PT-BR
python inference.py --visualize --sentence 'olá, vamos testar esse projeto.'
o|l|a| |,| |v|a|m|ʊ|s| |t|e|s|t|a| |e|s|i| |p|ɾ|o|ʒ|e|t|ʊ| |.
```

Para verificar parâmetros opcionais implementados no script, como formato de saída (IPA / X-SAMPA) ou seleção de sotaque (`--sotaque`):

```bash
python inference.py --help
```

### Testes em Lote e de Sotaques

Para executar testes G2P em lote sobre listas de entrada (como `list.txt`), salvando os resultados em `results/resultado_<sotaque>.txt` (usando `spx` como sotaque padrão):

```bash
python batch_test.py
```

Para executar testes entre todos os sotaques/variantes disponíveis, exibindo a saída diretamente no terminal:

```bash
python test_all_accents.py
```

## Como Contribuir

Contribuições são bem-vindas de pesquisadores, linguistas e desenvolvedores interessados em G2P para o português brasileiro, sotaques regionais e recursos fonéticos. Você pode contribuir da seguinte forma:

1. Faça um *fork* do repositório no GitHub.
2. Crie uma *branch* para sua melhoria ou correção (`git checkout -b feature/sua-melhoria`).
3. Adicione ou aprimore dicionários de pronúncia, variantes regionais, scripts de avaliação ou documentação (`resources/`, `CSVs/`, `evaluation/`).


4. Abra uma **Issue** para relatar problemas, sugerir funcionalidades ou discutir melhorias nos dados/sotaques.
5. Envie um **Pull Request** com uma descrição clara das alterações e de como elas foram testadas.

## Perguntas Frequentes (FAQ)

* **O que significa ANG2P?**
**ANG2P** significa *Accent-aware Neural Grapheme-to-Phoneme* (Conversão Neural Grafema-Fonema Sensível a Sotaques), um modelo G2P neural para o português brasileiro projetado com foco em variação regional e interpretabilidade.


* **Por que suportar tanto IPA quanto X-SAMPA?**
O IPA é o alfabeto fonético padrão para descrição acústica e articulatória, enquanto mapeamentos baseados em ASCII, como o X-SAMPA, ajudam a evitar problemas de codificação/Unicode em ferramentas de linha de comando e pipelines legados de fala/PLN.


* **Posso treinar o ANG2P com meu próprio dicionário de pronúncia ou dados regionais?**
Sim. Você pode fornecer dicionários de pronúncia e recursos personalizados seguindo a estrutura de `resources/` e `CSVs/`, ajustando as configurações de treinamento antes de executar `python train.py`.


* **Como posso visualizar como o modelo alinha grafemas e fonemas?**
Execute `python inference.py --visualize --sentence "..."` para gerar as visualizações de atenção e inspecionar o comportamento de alinhamento do modelo.



## Referências e Créditos

Este projeto se baseia e referencia trabalhos e repositórios anteriores:

* [hajix/G2P](https://github.com/hajix/G2P) — Implementação de referência original de G2P neural baseado em atenção adaptada neste trabalho.


* [fabianoluzbr/neural-g2p-portuguese](https://github.com/fabianoluzbr/neural-g2p-portuguese) — Trabalho anterior e recursos de G2P neural para o português.



## Contato

Para dúvidas, colaborações de pesquisa ou problemas relacionados a este repositório:

| Função | Nome / Canal |
| --- | --- |
| **Repositório** | [steinbukken7321/neural-g2p-portuguese](https://github.com/steinbukken7321/neural-g2p-portuguese) |
| **Supervisor** | *(adicione o nome aqui)* |
| **Autor** | **Rafael Ziani de Carvalho** (`@steinbukken7321`) |

## Licença

Este projeto está licenciado sob a Licença MIT. Consulte o arquivo [LICENSE](https://www.google.com/search?q=LICENSE) para mais detalhes.