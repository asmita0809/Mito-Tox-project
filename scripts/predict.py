#!/usr/bin/env python3
"""
Command-line mitochondrial-toxicity screening.

    py -3 scripts/predict.py --smiles "CC(C)(C)c1cc(C)cc(C(C)(C)C)c1O"
    py -3 scripts/predict.py --input examples/example_input.csv \
                             --smiles-column smiles --name-column name \
                             --output results/tables/my_predictions.csv

Output columns
    input_smiles, name          as supplied
    standardized_smiles         the parent structure actually scored
    inchikey                    full InChIKey of that parent
    probability                 predicted probability of MMP disruption
    predicted_class             1 if probability >= threshold, else 0
    threshold                   0.625, fixed before the test set was scored
    applicability               within domain | borderline | outside domain
    nn_tanimoto                 similarity to the nearest training compound
    in_training_set             True if this compound was used to fit the model
    training_set_label          its experimental label, when it was
    scored                      False means the structure was refused
    warnings                    why, and anything else qualifying the number

A probability is a computational prioritisation, not evidence of toxicity.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mitotox.predict import MitoToxPredictor

COLS = ["input_smiles", "name", "standardized_smiles", "inchikey", "probability",
        "predicted_class", "threshold", "applicability", "nn_tanimoto",
        "in_training_set", "training_set_label", "scored", "applicability_note",
        "warnings"]


def main():
    ap = argparse.ArgumentParser(
        description="Predict mitochondrial membrane potential disruption from SMILES.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smiles", help="a single SMILES string")
    g.add_argument("--input", help="CSV file containing a SMILES column")
    ap.add_argument("--smiles-column", default="smiles")
    ap.add_argument("--name-column", default=None)
    ap.add_argument("--output", default=None, help="write results to this CSV")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    if args.smiles:
        smis, names = [args.smiles], [None]
    else:
        df = pd.read_csv(args.input)
        if args.smiles_column not in df.columns:
            sys.stderr.write("column %r not in %s (columns: %s)\n"
                             % (args.smiles_column, args.input, list(df.columns)))
            return 1
        smis = df[args.smiles_column].tolist()
        names = (df[args.name_column].tolist()
                 if args.name_column and args.name_column in df.columns
                 else [None] * len(smis))

    pred = MitoToxPredictor(model_path=args.model)
    out = pred.predict(smis, names)[COLS].sort_values(
        "probability", ascending=False, na_position="last")
    for c in ("predicted_class", "training_set_label"):
        out[c] = out[c].astype("Int64")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out.to_csv(args.output, index=False)
        print("written: %s  (%d rows)" % (args.output, len(out)))

    show = out[["name", "standardized_smiles", "probability", "predicted_class",
                "applicability", "nn_tanimoto", "in_training_set"]].copy()
    show["standardized_smiles"] = show.standardized_smiles.str.slice(0, 34)
    print()
    print(show.to_string(index=False))
    n_un = int((~out.scored).sum())
    if n_un:
        print("\n%d structure(s) were NOT scored:" % n_un)
        for _, r in out[~out.scored].iterrows():
            print("  %-28s %s" % (str(r["name"] or r.input_smiles)[:28], r.warnings))
    n_tr = int(out.in_training_set.sum())
    if n_tr:
        print("\n%d compound(s) are IN the training set - those rows are fitted "
              "values, not predictions." % n_tr)
    print("\nPredicted mitochondrial-toxicity risk is a computational "
          "prioritisation, not experimental evidence of toxicity.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
