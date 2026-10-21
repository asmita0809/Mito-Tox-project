#!/usr/bin/env python3
"""The contract test: src/mitotox/standardize.py must reproduce the training set.

predict.py standardises incoming SMILES with the same function that produced
data/processed/tox21_mmp_training.csv. If the two ever drift apart, predictions
become invalid silently - no metric would show it. This test re-derives every
stored parent structure from its stored depositor SMILES and demands an exact
string match.

Run:  py -3 tests/test_standardization.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mitotox.standardize import standardize

STRUCTURES = os.path.join("results", "checkpoint_1D_structures.csv")


def main():
    if not os.path.exists(STRUCTURES):
        sys.stderr.write("run from the project root; %s not found\n" % STRUCTURES)
        return 1
    df = pd.read_csv(STRUCTURES, dtype=str)
    df = df[df.std_ok.astype(str).str.lower().isin(["true", "1"])]

    # The 6 metallocenes are expected to fail: their parent reduces to the
    # unparseable 'c1cccc1'. They carry flag_parent_unparseable and were
    # excluded from the dataset, so standardize() rejecting them up front is
    # the correct behaviour, not a regression.
    expected_fail = set(
        df.loc[~df.parent_parses.astype(str).str.lower()
               .isin(["true", "1"]), "pubchem_cid"])

    mism_smiles, mism_key, failed = [], [], []
    for _, r in df.iterrows():
        res = standardize(r.depositor_smiles)
        if not res["ok"]:
            failed.append((r.pubchem_cid, res["reason"]))
            continue
        if res["parent_smiles"] != r.parent_neutral_smiles:
            mism_smiles.append((r.pubchem_cid, r.parent_neutral_smiles,
                                res["parent_smiles"]))
        if res["inchikey"] != r.parent_neutral_inchikey:
            mism_key.append((r.pubchem_cid, r.parent_neutral_inchikey,
                             res["inchikey"]))

    got_fail = {c for c, _ in failed}
    unexpected = sorted(got_fail - expected_fail)
    missing = sorted(expected_fail - got_fail)

    print("structures checked              : %d" % len(df))
    print("expected failures (metallocenes): %d" % len(expected_fail))
    print("actual failures                 : %d" % len(failed))
    print("UNEXPECTED failures             : %d" % len(unexpected))
    print("expected failures that passed   : %d" % len(missing))
    print("parent SMILES mismatches        : %d" % len(mism_smiles))
    print("InChIKey mismatches             : %d" % len(mism_key))
    for cid, want, got in mism_smiles[:5]:
        print("   CID %-10s stored %s" % (cid, want[:60]))
        print("   %14s got    %s" % ("", got[:60]))
    for cid in unexpected[:5]:
        print("   CID %-10s UNEXPECTED FAILURE" % cid)
    print("-" * 60)
    if unexpected or missing or mism_smiles or mism_key:
        print("FAIL - the prediction path would not match the training path.")
        return 1
    print("PASS - standardize() reproduces every stored parent structure and "
          "InChIKey exactly,\n       and rejects exactly the %d structures flagged "
          "unparseable. The prediction\n       path is identical to the training path."
          % len(expected_fail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
