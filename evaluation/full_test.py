#!/usr/bin/env python3
"""
Teste completo do projeto G2P: todas as regiões e todos os modos.

Rode na RAIZ do projeto (onde ficam inference.py, train.py, utils/...):

    python full_test.py                  # tudo, 1000 palavras por região
    python full_test.py --n 300          # mais rápido
    python full_test.py --regions spx dli
    python full_test.py --smoke-train spx   # (opcional) treina 1 época de teste

Saída: test_results/report.md, test_results/results.json e uma lista por região.

Atenção: as métricas de PER/WER são calculadas sobre palavras que o modelo
JÁ VIU no treino (ainda não existe conjunto de teste separado). São um teto
otimista: servem para achar regiões quebradas e erros de dados, não para
medir generalização.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

REGIONS = ["lda", "rjo", "rjx", "spx", "spo", "map", "mpx", "lbx", "lbn", "dli"]
SENTENCE = "olá, vamos testar esse projeto."
EPOCHS = int(os.getenv("EPOCHS", "30"))
OUT = Path("test_results")

# Palavras para comparar sotaques (palatalização, rótico, /s/ em coda, nasais,
# palavras fora do léxico, estrangeirismos).
WORDS = [
    "tia", "dia", "leite", "carro", "rato", "porta", "sexta", "pasta",
    "bom", "mãe", "homem", "queijo", "filho", "xícara", "exceção",
    "hospital", "computador", "telecomunicações", "pneumonia", "trabalho",
    "internet", "whatsapp", "abacate",
]

# Casos de borda: devem falhar de forma controlada (ou ser tratados).
EDGE = {
    "maiúscula": "Olá",
    "hífen": "guarda-chuva",
    "dígitos": "123",
    "vazia": "",
    "NFD (a + til)": unicodedata.normalize("NFD", "ação"),
    "repetição": "z" * 20,
    "espaço": "a b",
}


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def run(cmd, env=None, timeout=900):
    t0 = time.time()
    e = dict(os.environ, **(env or {}))
    e["PYTHONIOENCODING"] = "utf-8"
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, env=e)
        return {"rc": p.returncode, "secs": round(time.time() - t0, 1),
                "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"rc": -1, "secs": timeout, "stdout": "", "stderr": "TIMEOUT"}


def last_line(text):
    lines = [l for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else ""


# ----------------------------------------------------------------------------
# Worker: carrega o modelo UMA vez e avalia uma região (roda em subprocesso,
# porque utils/config.py lê a variável SOTAQUE no momento do import)
# ----------------------------------------------------------------------------
def worker(region, n, out_json):
    os.environ["SOTAQUE"] = region
    sys.path.insert(0, os.getcwd())
    import torch
    from model import Encoder, Decoder
    from utils.data import ParserLexicon
    from utils.config import DataConfig, ModelConfig, TestConfig

    ds = ParserLexicon(DataConfig.graphemes_path, DataConfig.phonemes_path,
                       DataConfig.lexicon_path)
    enc_m = Encoder(ModelConfig.graphemes_size, ModelConfig.hidden_size)
    dec_m = Decoder(ModelConfig.phonemes_size, ModelConfig.hidden_size)
    enc_m.load_state_dict(torch.load(TestConfig.encoder_model_path, map_location="cpu"))
    dec_m.load_state_dict(torch.load(TestConfig.decoder_model_path, map_location="cpu"))
    enc_m.eval()
    dec_m.eval()

    def decode(word):
        x = torch.tensor([0] + [ds.g2idx[c] for c in word] + [1]).long().unsqueeze(1)
        phones = []
        with torch.no_grad():
            enc = enc_m(x)
            tok = torch.zeros(1, 1).long()
            hidden = torch.ones(1, 1, ModelConfig.hidden_size)
            for _ in range(len(word) * 3 + 10):
                out, hidden, _ = dec_m(tok, enc, hidden)
                idx = out[0, 0].argmax().item()
                if idx == 1:
                    return phones, True
                phones.append(ds.idx2p[idx])
                tok = torch.tensor([[idx]])
        return phones, False  # estourou o limite sem emitir <eos>

    refs = {}
    for w, p in ds.lexicon:
        refs.setdefault(w, set()).add(p)

    phone_freq = Counter(ph for _, p in ds.lexicon for ph in p.split())
    sample = random.Random(0).sample(list(refs), min(n, len(refs)))

    err_ph = tot_ph = exact = no_eos = 0
    errors = []
    for w in sample:
        pred, ended = decode(w)
        ref = min(refs[w], key=lambda r: edit_distance(pred, r.split())).split()
        d = edit_distance(pred, ref)
        err_ph += d
        tot_ph += len(ref)
        exact += d == 0
        no_eos += not ended
        if d:
            errors.append((d, w, " ".join(pred), " ".join(ref)))
    errors.sort(reverse=True)

    words_out = {}
    for w in WORDS:
        pred, ended = decode(w)
        words_out[w] = {"pred": " ".join(pred), "in_lexicon": w in refs,
                        "ref": sorted(refs.get(w, []))}

    edge_out = {}
    for name, w in EDGE.items():
        try:
            pred, ended = decode(w)
            edge_out[name] = "OK: " + " ".join(pred) + ("" if ended else " [sem <eos>]")
        except Exception as e:  # noqa: BLE001
            edge_out[name] = f"ERRO {type(e).__name__}: {e}"

    homographs = sum(len(v) > 1 for v in refs.values())
    result = {
        "lexicon_entries": len(ds.lexicon),
        "unique_words": len(refs),
        "homograph_words": homographs,
        "tested": len(sample),
        "PER": round(100 * err_ph / max(tot_ph, 1), 2),
        "WER": round(100 * (1 - exact / max(len(sample), 1)), 2),
        "no_eos": no_eos,
        "rare_phonemes": {p: c for p, c in sorted(phone_freq.items(), key=lambda x: x[1]) if c < 20},
        "worst_errors": errors[:25],
        "words": words_out,
        "edge": edge_out,
    }
    Path(out_json).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------
# Orquestrador
# ----------------------------------------------------------------------------
def test_region(region, n):
    r = {"region": region}
    enc = f"checkpoints/{region}/encoder_e{EPOCHS:02}.pth"
    dec = f"checkpoints/{region}/decoder_e{EPOCHS:02}.pth"
    lex = f"resources/lexicons/{region}.json"
    r["files"] = {p: os.path.exists(p) for p in (enc, dec, lex)}
    if not all(r["files"].values()):
        r["error"] = "arquivos ausentes: " + ", ".join(p for p, ok in r["files"].items() if not ok)
        return r

    py = sys.executable

    # inference.py: ipa, xsampa, visualize
    ipa = run([py, "inference.py", "--sotaque", region, "--format", "ipa", "--sentence", SENTENCE])
    xs = run([py, "inference.py", "--sotaque", region, "--format", "xsampa", "--sentence", SENTENCE])
    viz = run([py, "inference.py", "--sotaque", region, "--visualize", "--sentence", SENTENCE])
    ipa_out, xs_out = last_line(ipa["stdout"]), last_line(xs["stdout"])
    r["cli"] = {
        "ipa": {"rc": ipa["rc"], "out": ipa_out, "err": ipa["stderr"][-300:]},
        "xsampa": {"rc": xs["rc"], "out": xs_out, "err": xs["stderr"][-300:]},
        "visualize": {"rc": viz["rc"], "png": os.path.exists(f"attention/{region}/vamos.png"),
                      "err": viz["stderr"][-300:]},
        "xsampa_same_len": len(ipa_out.split("|")) == len(xs_out.split("|")),
        "xsampa_non_ascii": sorted({c for c in xs_out if ord(c) > 127}),
        "secs_per_call": ipa["secs"],
    }

    # batch_test.py
    lst = OUT / "list.txt"
    lst.write_text("\n".join(WORDS), encoding="utf-8")
    outp = OUT / f"batch_{region}.txt"
    b = run([py, "batch_test.py", "--sotaque", region, "--list_path", str(lst),
             "--output_path", str(outp)])
    n_lines = len(outp.read_text(encoding="utf-8").splitlines()) if outp.exists() else 0
    r["batch"] = {"rc": b["rc"], "lines": n_lines, "expected": len(WORDS), "err": b["stderr"][-300:]}

    # avaliação quantitativa + palavras + bordas (modelo carregado uma vez)
    wj = OUT / f"worker_{region}.json"
    w = run([py, __file__, "--worker", region, "--n", str(n), "--worker-out", str(wj)], timeout=3600)
    if w["rc"] == 0 and wj.exists():
        r["eval"] = json.loads(wj.read_text(encoding="utf-8"))
    else:
        r["eval_error"] = w["stderr"][-500:]
    return r


def write_report(results):
    L = ["# Relatório de teste G2P", "",
         "> PER/WER medidos em palavras do próprio treino (sem conjunto de teste): teto otimista.", ""]
    L += ["## Resumo por região", "",
          "| região | entradas | testadas | PER % | WER % | sem <eos> | homógrafos | ipa | xsampa | viz | batch |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    ok = lambda b: "ok" if b else "FALHA"  # noqa: E731
    for r in results:
        if "error" in r:
            L.append(f"| {r['region']} | — | — | — | — | — | — | {r['error']} | | | |")
            continue
        e, c, b = r.get("eval", {}), r["cli"], r["batch"]
        L.append(f"| {r['region']} | {e.get('lexicon_entries', '?')} | {e.get('tested', '?')} | "
                 f"{e.get('PER', '?')} | {e.get('WER', '?')} | {e.get('no_eos', '?')} | "
                 f"{e.get('homograph_words', '?')} | {ok(c['ipa']['rc'] == 0)} | "
                 f"{ok(c['xsampa']['rc'] == 0 and c['xsampa_same_len'] and not c['xsampa_non_ascii'])} | "
                 f"{ok(c['visualize']['rc'] == 0 and c['visualize']['png'])} | "
                 f"{ok(b['rc'] == 0 and b['lines'] == b['expected'])} |")

    good = [r for r in results if "eval" in r]
    L += ["", f"## Frase de teste: `{SENTENCE}`", ""]
    for r in good:
        L += [f"**{r['region']}**", "", f"- IPA: `{r['cli']['ipa']['out']}`",
              f"- X-SAMPA: `{r['cli']['xsampa']['out']}`"]
        if r["cli"]["xsampa_non_ascii"]:
            L.append(f"- ⚠ símbolos sem mapeamento X-SAMPA: {r['cli']['xsampa_non_ascii']}")
        L.append("")

    L += ["## Comparação entre sotaques (mesmas palavras)", "",
          "| palavra | " + " | ".join(r["region"] for r in good) + " |",
          "|---|" + "---|" * len(good)]
    for w in WORDS:
        cells = []
        for r in good:
            d = r["eval"]["words"][w]
            cells.append(d["pred"].replace(" ", "") + ("" if d["in_lexicon"] else " *"))
        L.append(f"| {w} | " + " | ".join(cells) + " |")
    L += ["", "`*` = palavra fora do léxico daquela região (generalização real).", ""]

    L += ["## Casos de borda", ""]
    for r in good:
        L.append(f"**{r['region']}**")
        L += [f"- {k}: {v}" for k, v in r["eval"]["edge"].items()]
        L.append("")

    L += ["## Fonemas raros (<20 ocorrências no léxico)", ""]
    for r in good:
        L.append(f"- {r['region']}: {r['eval']['rare_phonemes'] or 'nenhum'}")

    L += ["", "## Piores erros (palavra | previsto | referência)", ""]
    for r in good:
        L += [f"**{r['region']}**", ""]
        L += [f"- {w} | {p} | {ref}" for _, w, p, ref in r["eval"]["worst_errors"][:10]]
        L.append("")

    bad = [r for r in results if "eval_error" in r or any(
        r.get("cli", {}).get(k, {}).get("rc", 0) for k in ("ipa", "xsampa", "visualize"))
        or r.get("batch", {}).get("rc", 0)]
    if bad:
        L += ["## Erros de execução", ""]
        for r in bad:
            L.append(f"**{r['region']}**")
            for k in ("ipa", "xsampa", "visualize"):
                if r.get("cli", {}).get(k, {}).get("rc"):
                    L.append(f"- {k}: {r['cli'][k]['err']}")
            if r.get("batch", {}).get("rc"):
                L.append(f"- batch: {r['batch']['err']}")
            if "eval_error" in r:
                L.append(f"- eval: {r['eval_error']}")
            L.append("")
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", default=REGIONS)
    ap.add_argument("--n", type=int, default=1000, help="palavras avaliadas por região")
    ap.add_argument("--smoke-train", metavar="REGIAO", help="treina 1 época (EPOCHS=1) como teste")
    ap.add_argument("--worker")
    ap.add_argument("--worker-out")
    a = ap.parse_args()

    if a.worker:
        worker(a.worker, a.n, a.worker_out)
        return

    OUT.mkdir(exist_ok=True)
    results = []
    for region in a.regions:
        print(f"\n=== {region} ===", flush=True)
        r = test_region(region, a.n)
        results.append(r)
        e = r.get("eval")
        print("  " + (r.get("error") or r.get("eval_error") or
                      f"PER {e['PER']}% | WER {e['WER']}% | sem <eos>: {e['no_eos']}"), flush=True)

    if a.smoke_train:
        print(f"\n=== smoke train {a.smoke_train} (1 época) ===")
        t = run([sys.executable, "train.py", "--sotaque", a.smoke_train], env={"EPOCHS": "1"}, timeout=7200)
        print("  rc =", t["rc"], "|", t["stderr"][-300:])

    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(results)
    print(f"\nRelatório: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()