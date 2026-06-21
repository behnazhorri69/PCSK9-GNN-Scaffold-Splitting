import pandas as pd
import numpy as np
import random
from collections import defaultdict

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina


# ------------------------------
# Fingerprint generation
# ------------------------------

def smiles_to_fps(smiles, radius=2, nbits=2048):

    mols = [Chem.MolFromSmiles(s) for s in smiles]

    fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
        for m in mols
    ]

    return fps


# ------------------------------
# Sphere exclusion split
# ------------------------------

def sphere_exclusion_split(fps, threshold=0.35):

    selected = []
    remaining = list(range(len(fps)))

    while remaining:

        i = remaining.pop(0)
        selected.append(i)

        new_remaining = []

        for j in remaining:
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            if sim < threshold:
                new_remaining.append(j)

        remaining = new_remaining

    return selected


# ------------------------------
# Main split function
# ------------------------------

def molecular_dataset_split(
    csv_file,
    smiles_col="smiles",
    fp_col=None,
    method="scaffold",
    frac_train=0.8,
    frac_val=0.1,
    frac_test=0.1,
    cluster_cutoff=0.3,
    sphere_threshold=0.35,
    seed=42
):

    random.seed(seed)

    df = pd.read_excel(csv_file)

    smiles = df[smiles_col].tolist()
    n_total = len(smiles)

    train_cut = int(frac_train * n_total)
    val_cut = int(frac_val * n_total)

    train_idx, val_idx, test_idx = [], [], []

    # -----------------------
    # RANDOM SPLIT
    # -----------------------

    if method == "random":

        indices = list(range(n_total))
        random.shuffle(indices)

        train_idx = indices[:train_cut]
        val_idx = indices[train_cut:train_cut+val_cut]
        test_idx = indices[train_cut+val_cut:]


    # -----------------------
    # SCAFFOLD SPLIT
    # -----------------------

    elif method == "scaffold":

        scaffolds = defaultdict(list)

        for i, smi in enumerate(smiles):

            mol = Chem.MolFromSmiles(smi)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

            scaffolds[scaffold].append(i)

        groups = sorted(scaffolds.values(), key=len, reverse=True)
        np.random.shuffle(groups)

        for g in groups:

            if len(train_idx) + len(g) <= train_cut:
                train_idx.extend(g)

            elif len(val_idx) + len(g) <= val_cut:
                val_idx.extend(g)

            else:
                test_idx.extend(g)


    # -----------------------
    # CLUSTER SPLIT
    # -----------------------

    elif method == "cluster":

        fps = smiles_to_fps(smiles)

        dists = []

        for i in range(1, len(fps)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1-x for x in sims])

        clusters = Butina.ClusterData(
            dists,
            len(fps),
            cluster_cutoff,
            isDistData=True
        )

        clusters = sorted(clusters, key=len, reverse=True)

        for c in clusters:

            c = list(c)

            if len(train_idx) + len(c) <= train_cut:
                train_idx.extend(c)

            elif len(val_idx) + len(c) <= val_cut:
                val_idx.extend(c)

            else:
                test_idx.extend(c)


    # -----------------------
    # SPHERE EXCLUSION
    # -----------------------

    elif method == "sphere":

        fps = smiles_to_fps(smiles)

        diverse_idx = sphere_exclusion_split(
            fps,
            threshold=sphere_threshold
        )

        remaining = list(set(range(n_total)) - set(diverse_idx))

        indices = diverse_idx + remaining

        train_idx = indices[:train_cut]
        val_idx = indices[train_cut:train_cut+val_cut]
        test_idx = indices[train_cut+val_cut:]


    else:
        raise ValueError("method must be random, scaffold, cluster, or sphere")


    # -----------------------
    # SAVE FILES
    # -----------------------

    df.iloc[train_idx].to_csv("train.csv", index=False)
    df.iloc[val_idx].to_csv("valid.csv", index=False)
    df.iloc[test_idx].to_csv("test.csv", index=False)

    print("Train:", len(train_idx))
    print("Valid:", len(val_idx))
    print("Test:", len(test_idx))
    
    return train_idx, val_idx, test_idx


# ------------------------------
# Example run
# ------------------------------

if __name__ == "__main__":

    molecular_dataset_split(
        "molecules.csv",
        smiles_col="SMILES",
        method="scaffold"
    )
