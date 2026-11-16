"""Single source of truth for every constant in the pipeline.

Anything that would otherwise be duplicated across scripts lives here so the
numbers cannot drift. Imported by features.py, interpretation.py and the
scripts/ entry points.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "external"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

TRAINING_CSV = DATA_PROCESSED / "tox21_mmp_training.csv"

# Named artefacts, so that a script and the file it reads cannot drift apart.
# These paths were previously spelled out as string literals in three scripts,
# and when the outputs were reorganised into results/tables/ the literals were
# not updated: final_test_eval.py, select_model.py and train_models.py then
# pointed at files that did not exist, so the documented run order failed on a
# clean checkout. Every path below now has exactly one definition.
SPLITS_CSV = DATA_PROCESSED / "splits.csv"
DESCRIPTORS_CSV = DATA_PROCESSED / "features_descriptors.csv"
MORGAN_NPZ = DATA_PROCESSED / "features_morgan.npz"
MACCS_NPZ = DATA_PROCESSED / "features_maccs.npz"
TEST_NN_SIMILARITY_CSV = DATA_PROCESSED / "test_nn_similarity.csv"
OOF_PREDICTIONS_NPZ = DATA_PROCESSED / "oof_predictions_dev.npz"

CV_RESULTS_CSV = TABLES / "cv_results.csv"
CV_RESULTS_RANKED_CSV = TABLES / "cv_results_ranked.csv"
TEST_PREDICTIONS_CSV = TABLES / "test_predictions.csv"
TEST_BY_PROXIMITY_CSV = TABLES / "test_performance_by_proximity.csv"
MODEL_SELECTION_JSON = RESULTS / "model_selection.json"
FINAL_TEST_RESULTS_JSON = RESULTS / "final_test_results.json"

# ---------------------------------------------------------------- reproducibility
RANDOM_SEED = 42

# ---------------------------------------------------------------- fingerprints
# Morgan / ECFP-like circular fingerprint.
#   radius 2  -> ECFP4 equivalent; the standard QSAR default. Radius 3 doubles
#                the substructure vocabulary without usually improving small
#                datasets, and makes bit interpretation harder.
#   2048 bits -> collision rate is low for drug-sized molecules while keeping
#                the matrix small enough to model on 5,938 compounds.
#   binary    -> presence/absence, not counts. Counts add scale sensitivity that
#                tree models do not need and logistic regression handles badly
#                without extra scaling.
#   chirality -> ON, because the curated set deliberately keeps stereoisomers as
#                separate compounds (full InChIKey grouping). Ignoring chirality
#                here would give two distinct compounds identical features.
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
MORGAN_BINARY = True
MORGAN_USE_CHIRALITY = True

# MACCS: 167 predefined structural keys (index 0 is unused by RDKit).
MACCS_NBITS = 167

# ---------------------------------------------------------------- split grouping
# Analogue grouping for the train/test split. Locked at Checkpoint 3A.
# Bemis-Murcko was rejected on evidence: 1,167 compounds share the scaffold
# 'c1ccccc1' (19.7% of the dataset) and 1,458 are acyclic with no scaffold at
# all, which made StratifiedGroupKFold degenerate (a 34-compound test set).
SPLIT_GROUPING = "butina"
BUTINA_CUTOFF = 0.30          # DISTANCE; 0.30 == Tanimoto similarity 0.70
N_OUTER_FOLDS = 5             # fold 0 becomes the frozen test set
N_CV_FOLDS = 5                # cross-validation within the development set

# ---------------------------------------------------------------- descriptors
# Exactly the six named in the project brief. No more: every extra descriptor is
# another dimension on a 5,938-row dataset and another thing to justify.
DESCRIPTORS = [
    ("MolWt", "Average molecular weight (Da)"),
    ("MolLogP", "Wildman-Crippen octanol/water partition coefficient (logP)"),
    ("TPSA", "Topological polar surface area (A^2), Ertl method"),
    ("NumHDonors", "Lipinski hydrogen-bond donor count"),
    ("NumHAcceptors", "Lipinski hydrogen-bond acceptor count"),
    ("NumRotatableBonds", "Number of rotatable bonds (RDKit strict definition)"),
]

# ---------------------------------------------------------------- feature policy
# Structural alerts are POST-HOC INTERPRETATION ONLY. They are never
# concatenated into the model feature matrix. See interpretation.py.
ALERTS_ARE_MODEL_FEATURES = False
