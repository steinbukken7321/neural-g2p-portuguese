import os
import csv
import json
import glob
import unicodedata


# ============================================================
# CONSTANTES
# ============================================================

# Marcadores de prosódia/silabação a remover da transcrição fonética
PROSODY_CHARS = {".", ",", "'", "ˌ", "ˈ", "·"}

# Caracteres soltos que aparecem como ruído/erro de origem e não
# representam fonemas reais:
#   "-"  -> hífen de prefixo (ex: "in-operacionalidade") que vazou
#           para dentro da transcrição fonética
#   "~"  -> til ASCII solto e redundante ao lado de uma vogal que
#           já carrega nasalização própria (ex: "kõ~.nˈoʃ.kʊ",
#           onde "õ" já é nasal e o "~" extra é lixo)
NOISE_CHARS = {"-", "~"}

# Padrões que indicam a linha inteira corrompida na fonte, sem
# possibilidade de recuperar um fonema válido dela:
#   "collins"/"ollins" -> sobra de nota do tipo "Collins não tem"
#                          que grudou na transcrição por bug no
#                          scraping da fonte original
#   "_"                -> tokens quebrados/transcrição corrompida
#                          (ex: "ˈar_rr", "dɨ.zi_s.pəɾ.ti.ʎˈaɾ")
INVALID_PATTERNS = ("collins", "ollins", "_")

# Delimitadores de pronúncias alternativas encontrados na fonte.
# Nesses casos ficamos apenas com a PRIMEIRA variante, em vez de
# descartar a palavra inteira.
ALT_PRONUNCIATION_DELIMITERS = ("$$", " ou ")


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
# EXTRAÇÃO DA PRONÚNCIA PRIMÁRIA
# ============================================================

def extract_primary_pronunciation(raw_phonetic):
    """
    Alguns registros na fonte trazem mais de uma pronúncia
    alternativa para a mesma palavra, separadas por '$$' ou pela
    palavra ' ou ' (ex: "vˈaɽ.gə ou vˈaɽ.gə",
    "bˈɔɾ.du$$bˈoɾ.dʊ"). Nesses casos, ficamos só com a primeira
    variante em vez de descartar a entrada inteira.
    """

    if not isinstance(raw_phonetic, str):
        return raw_phonetic

    result = raw_phonetic
    for delimiter in ALT_PRONUNCIATION_DELIMITERS:
        if delimiter in result:
            result = result.split(delimiter, 1)[0]

    return result.strip()


# ============================================================
# DETECÇÃO DE ENTRADA CORROMPIDA
# ============================================================

def is_corrupted(raw_phonetic):
    """
    Detecta transcrições fonéticas com ruído conhecido de origem
    que não pode ser limpo, só descartado. Deve ser chamada sobre
    o texto já processado por extract_primary_pronunciation.
    """

    if not raw_phonetic:
        return False

    lowered = raw_phonetic.lower()

    return any(pattern in lowered for pattern in INVALID_PATTERNS)


# ============================================================
# DIACRÍTICOS IPA QUE NÃO SÃO "COMBINING MARKS" UNICODE
# ============================================================

# ʰ (aspirado), ʷ (labializado) e ː (longo) são diacríticos/
# suprassegmentais legítimos do IPA (ver quadro oficial da IPA),
# mas tecnicamente são "Spacing Modifier Letters" no Unicode, não
# "combining marks" - por isso unicodedata.combining() não os
# reconhece e clean_phonetic() os trataria como fonemas soltos e
# independentes em vez de anexá-los ao fonema anterior (ex: "t" e
# "ʰ" separados, quando deveriam formar um único fonema "tʰ",
# representando a consoante aspirada).
IPA_SPACING_MODIFIERS = {"ʰ", "ʷ", "ː"}


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

    # Caracteres que devem ser removidos: marcadores de prosódia
    # + ruído conhecido de origem (hífen de prefixo, til solto)
    chars_to_remove = PROSODY_CHARS | NOISE_CHARS

    # Remove espaços existentes para reconstruir
    # a separação padronizada posteriormente
    normalized = "".join(
        char
        for char in normalized
        if char not in chars_to_remove and not char.isspace()
    )

    phonemes = []

    for char in normalized:

        # Caracteres combinantes (til, acento etc.) e os
        # diacríticos IPA de IPA_SPACING_MODIFIERS (que o Unicode
        # não classifica como "combining", mas que na prática
        # modificam o fonema anterior) são anexados ao fonema
        # anterior em vez de virarem um token à parte
        if unicodedata.combining(char) or char in IPA_SPACING_MODIFIERS:

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
    total_corrompidos_geral = 0
    total_multiplas_palavras_geral = 0

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
        corrompidos_regiao = 0
        multiplas_palavras_regiao = 0

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

                # Fica só com a primeira variante quando há
                # pronúncias alternativas na mesma célula
                raw_phone = extract_primary_pronunciation(raw_phone)

                # Descarta entradas com ruído conhecido de
                # origem que não dá para limpar, só descartar
                if is_corrupted(raw_phone):
                    corrompidos_regiao += 1
                    continue

                # Limpeza
                word = clean_word(raw_word)
                phonetic = clean_phonetic(raw_phone)

                # Descarta expressões/locuções com mais de uma
                # palavra (ex: "água de colônia"). O vocabulário
                # de grafemas (Graphemes.json) não inclui espaço,
                # pois o modelo é treinado para operar palavra a
                # palavra, não frase a frase.
                if " " in word:
                    multiplas_palavras_regiao += 1
                    continue

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
        total_corrompidos_geral += corrompidos_regiao
        total_multiplas_palavras_geral += multiplas_palavras_regiao

        # ====================================================
        # SALVA JSON
        # ====================================================

        with open(
            output_json_path,
            mode="w",
            encoding="utf-8"
        ) as out_f:

            out_f.write("[")

            for i, entry in enumerate(lexicon_entries):

                json_line = json.dumps(
                    entry,
                    ensure_ascii=False
                )

                if i < len(lexicon_entries) - 1:

                    out_f.write(
                        f"{json_line},\n"
                    )

                else:

                    out_f.write(
                        f"{json_line}"
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
            f"   • Registros nulos/vazios ignorados: "
            f"{nulos_regiao}"
        )

        print(
            f"   • Registros corrompidos descartados: "
            f"{corrompidos_regiao}"
        )

        print(
            f"   • Expressões/locuções descartadas (múltiplas palavras): "
            f"{multiplas_palavras_regiao}"
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
        f"Total de registros nulos/vazios encontrados: "
        f"{total_nulos_geral}"
    )

    print(
        f"Total de registros corrompidos descartados: "
        f"{total_corrompidos_geral}"
    )

    print(
        f"Total de expressões/locuções descartadas (múltiplas palavras): "
        f"{total_multiplas_palavras_geral}"
    )

    print(
        "================================================="
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    process_lexicon_csvs()