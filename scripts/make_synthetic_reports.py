import argparse
import random
import os
from pathlib import Path

# Sample HPO terms, genes, and variants
HPO = [
    ("HP:0004322", "Short stature"),
    ("HP:0001250", "Seizures"),
    ("HP:0002721", "Recurrent infections"),
    ("HP:0002027", "Diarrhea"),
    ("HP:0001537", "Abnormality of hair morphology"),
]

GENES = ["CFTR", "BRCA1", "BRCA2", "FBN1", "PAH", "PKD1", "COL1A1"]
VARIANTS = ["c.1521_1523delCTT (p.Phe508del)", "c.68_69delAG", "c.35delG", "c.1582G>A"]

TEMPLATE = (
    "Patient exhibits {hpo1}, {hpo2}. "
    "Genetic testing reveals a variant in {gene}: {var}. "
    "Family history is notable. Recommend correlation with phenotype and ACMG/AMP criteria."
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic de-identified genetic reports.")
    ap.add_argument("--out", default="data/toy_reports", help="Output directory for reports")
    ap.add_argument("--n", type=int, default=20, help="Number of synthetic reports to generate")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    for i in range(1, args.n + 1):
        gene = random.choice(GENES)
        var = random.choice(VARIANTS)
        hpo1, hpo2 = random.sample([h[1] for h in HPO], 2)
        text = TEMPLATE.format(hpo1=hpo1, hpo2=hpo2, gene=gene, var=var)
        (Path(args.out) / f"report_{i:03d}.txt").write_text(text, encoding="utf-8")

    print(f"Wrote {args.n} synthetic reports to {args.out}")
