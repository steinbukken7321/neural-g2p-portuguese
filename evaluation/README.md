# Módulo de Avaliação (Evaluation)

Esta pasta contém a suíte de ferramentas para avaliação métrica, auditoria de desempenho e testes automatizados de integração dos modelos treinados para cada sotaque.

## Arquivos e Módulos

| Arquivo | Descrição |
| :--- | :--- |
| **`evaluate.py`** | Calcula as métricas quantitativas formais de desempenho para um sotaque específico (**PER** - *Phoneme Error Rate* e **Word Accuracy**). Gera um relatório em CSV (`eval_<sotaque>.csv`) contendo todas as divergências entre as predições e o léxico de referência. |
| **`full_test.py`** | Suíte de testes automatizada de cobertura total. Avalia todas as 10 regiões suportadas, analisa o comportamento do modelo em casos de borda (maiúsculas, hífens, dígitos, palavras vazias), identifica fonemas raros (<20 ocorrências) e gera relatórios consolidados em `test_results/report.md` e `results.json`. |
| **`test_all_accents.py`** | Runner rápido para verificação visual e validação de inferência contínua. Executa o modelo em subprocesso para todas as regiões em ambos os formatos de saída (**IPA** e **X-SAMPA**) utilizando uma frase de teste padrão. |

## Exemplos de Uso

```bash
# Avaliação métrica quantitativa para um sotaque (PER / Word Accuracy)
python evaluation/evaluate.py --sotaque spx

# Execução da suíte de testes completa em todas as regiões
python evaluation/full_test.py

# Validação rápida de inferência em lote por região (IPA e X-SAMPA)
python evaluation/test_all_accents.py