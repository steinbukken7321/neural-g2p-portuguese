import subprocess

regions = [
    "lda", "rjo", "rjx", "spx", "spo", 
    "map", "mpx", "lbx", "lbn", "dli"
]

sentence = "olá, vamos testar esse projeto."

formats = ["ipa", "xsampa"]

for fmt in formats:
    print("=" * 60)
    print(f" TESTANDO TODAS AS REGIÕES NO FORMATO: {fmt.upper()}")
    print("=" * 60)
    
    for region in regions:
        print(f"\n--- Região: {region} [{fmt.upper()}] ---")
        command = [
            "python", "inference.py", 
            "--sotaque", region, 
            "--format", fmt, 
            "--sentence", sentence
        ]
        # Executa diretamente repassando a saída para o terminal atual
        subprocess.run(command)

print("\n" + "=" * 60)