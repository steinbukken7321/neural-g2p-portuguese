import os
import csv
import json
import glob
import unicodedata


# ============================================================
# LIMPEZA DA PALAVRA
# ============================================================

def clean_word(word):
    """
    Remove separadores silábicos da palavra e normaliza espaços.
    Mantém letras e acentos.
    """

    if not isinstance(word, str):
        return ""

    word = word.strip()

    # Remove separador silábico e hífens
    word = word.replace("·", "")
    word = word.replace("-", "")

    return word.strip().lower()


# ============================================================
# LIMPEZA DA FONÉTICA
# ============================================================

def clean_phonetic(phonetic):
    """
    Remove marcadores indesejados da transcrição fonética
    e separa cada fonema por espaço.

    Caracteres Unicode combinados, como:
        ã, õ, ɐ̃

    permanecem unidos corretamente.
    """

    if not isinstance(phonetic, str):
        return ""

    # Remove espaços desnecessários no início e fim
    phonetic = phonetic.strip()

    if not phonetic:
        return ""

    # Normaliza para NFD para tratar corretamente diacríticos
    normalized = unicodedata.normalize("NFD", phonetic)

    # Caracteres que devem ser removidos
    chars_to_remove = {
        ".",
        ",",
        "'",
        "ˌ",
        "ˈ",
        "·"
    }

    # Remove espaços existentes para reconstruir
    # a separação padronizada posteriormente
    normalized = "".join(
        char
        for char in normalized
        if char not in chars_to_remove and not char.isspace()
    )

    phonemes = []

    for char in normalized:

        # Se for um caractere combinante (til, acento etc.)
        # adiciona ao fonema anterior
        if unicodedata.combining(char):

            if phonemes:
                phonemes[-1] += char

        else:
            phonemes.append(char)

    # Reconverte Unicode para formato normal
    phonemes = [
        unicodedata.normalize("NFC", phoneme)
        for phoneme in phonemes
    ]

    # Junta os fonemas com exatamente UM espaço
    return " ".join(phonemes).strip()


# ============================================================
# PROCESSAMENTO DOS CSVs
# ============================================================

def process_lexicon_csvs(
    data_dir="../data",
    output_dir="../resources/lexicons"
):

    # Permite executar tanto da raiz quanto de outra pasta
    if not os.path.exists(data_dir) and os.path.exists("data"):
        data_dir = "data"
        output_dir = "resources/lexicons"

    # Cria diretório de saída
    os.makedirs(output_dir, exist_ok=True)

    # Procura todos os CSVs do dicionário fonético
    csv_files = glob.glob(
        os.path.join(
            data_dir,
            "Dicionario_Fonetico_*.csv"
        )
    )

    regioes_processadas = 0
    total_nulos_geral = 0

    # ========================================================
    # PROCESSA CADA CSV
    # ========================================================

    for csv_path in csv_files:

        filename = os.path.basename(csv_path)

        region_code = (
            filename
            .replace("Dicionario_Fonetico_", "")
            .replace(".csv", "")
            .lower()
        )

        output_json_path = os.path.join(
            output_dir,
            f"{region_code}.json"
        )

        # ----------------------------------------------------
        # NÃO SOBRESCREVE ARQUIVOS EXISTENTES
        # ----------------------------------------------------

        # if os.path.exists(output_json_path):
        #     print(
        #         f"Pulando {region_code}: "
        #         f"o arquivo {output_json_path} já existe."
        #     )
        #     continue

        lexicon_entries = []
        seen = set()

        nulos_regiao = 0

        print(
            f"\nProcessando nova região: "
            f"{region_code} ({filename})..."
        )

        # utf-8-sig remove corretamente BOM caso exista
        with open(
            csv_path,
            mode="r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            # Detecta delimitador
            sample = f.readline()

            f.seek(0)

            delimiter = ";" if ";" in sample else ","

            reader = csv.DictReader(
                f,
                delimiter=delimiter
            )

            # ------------------------------------------------
            # NORMALIZA NOMES DAS COLUNAS
            # ------------------------------------------------

            if reader.fieldnames:

                reader.fieldnames = [
                    fn.strip().lower()
                    for fn in reader.fieldnames
                ]

            # Procura coluna Palavra
            word_col = (
                next(
                    (
                        col
                        for col in reader.fieldnames
                        if "palavra" in col
                    ),
                    "palavra"
                )
                if reader.fieldnames
                else "palavra"
            )

            # Procura coluna Fonética/Fonetica
            phone_col = (
                next(
                    (
                        col
                        for col in reader.fieldnames
                        if (
                            "fonética" in col
                            or "fonetica" in col
                        )
                    ),
                    "fonética"
                )
                if reader.fieldnames
                else "fonética"
            )

            # =================================================
            # PROCESSA LINHAS
            # =================================================

            for row in reader:

                raw_word = row.get(word_col, "")
                raw_phone = row.get(phone_col, "")

                # Ignora valores nulos
                if (
                    not raw_word
                    or not raw_phone
                    or not str(raw_word).strip()
                    or not str(raw_phone).strip()
                ):
                    nulos_regiao += 1
                    continue

                # Limpeza
                word = clean_word(raw_word)
                phonetic = clean_phonetic(raw_phone)

                # Validação após limpeza
                if not word or not phonetic:
                    nulos_regiao += 1
                    continue

                entry = [
                    word,
                    phonetic
                ]

                tuple_entry = (
                    word,
                    phonetic
                )

                # Evita duplicatas
                if tuple_entry not in seen:

                    seen.add(tuple_entry)
                    lexicon_entries.append(entry)

        total_nulos_geral += nulos_regiao

        # ====================================================
        # SALVA JSON
        # ====================================================

        with open(
            output_json_path,
            mode="w",
            encoding="utf-8"
        ) as out_f:

            out_f.write("[\n")

            for i, entry in enumerate(lexicon_entries):

                json_line = json.dumps(
                    entry,
                    ensure_ascii=False
                )

                if i < len(lexicon_entries) - 1:

                    out_f.write(
                        f"  {json_line},\n"
                    )

                else:

                    out_f.write(
                        f"  {json_line}\n"
                    )

            out_f.write("]\n")

        regioes_processadas += 1

        print(
            f"-> Região '{region_code}' "
            f"processada com sucesso."
        )

        print(
            f"   • Palavras únicas encontradas: "
            f"{len(lexicon_entries)}"
        )

        print(
            f"   • Registros nulos/inválidos ignorados: "
            f"{nulos_regiao}"
        )

        print(
            f"   • Arquivo salvo em: "
            f"{output_json_path}"
        )

    # ========================================================
    # RELATÓRIO FINAL
    # ========================================================

    print(
        "\n================ RELATÓRIO FINAL ================"
    )

    print(
        f"Total de novas regiões processadas: "
        f"{regioes_processadas}"
    )

    print(
        f"Total de registros nulos/inválidos encontrados: "
        f"{total_nulos_geral}"
    )

    print(
        "================================================="
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    process_lexicon_csvs()