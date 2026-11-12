"""Extra checks on the feature search: significance, a predictive model, and robustness to how "extreme" is defined.

The main feature search (trait_confirmation.py) compares extreme NFAs with the rest. This script asks three more questions
of the same data (data/experiment_10k/):

  1. Are the trait differences statistically significant?  (Mann-Whitney U test per trait)
  2. How well do the traits predict which NFAs are extreme?  (random forest, 5-fold cross-validated AUC, and which traits
     the forest relies on, by permutation importance)
  3. Do the same traits stand out if "extreme" means the top 1 % instead of 5 %, or if the raw subset-construction DFA is
     used instead of the minimized DFA?

Needs scikit-learn and scipy in addition to the project's usual packages. Output: data/experiment_10k/feature_search_extras/.

    cd src && python -m experiments.feature_search_extras
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from experiments.blowup_experiment_10k import EXPERIMENT_10K_DIR
from experiments.trait_confirmation import TRAIT_COLUMNS, load_features

# Two extra ratios given to the forest on top of the 24 traits
FOREST_EXTRA_COLUMNS = ["trans_per_state", "det_slot_frac"]


def add_ratios(df):
    df = df.copy()
    df["trans_per_state"] = df["num_symbol_transitions"] / df["original_states"]
    # Share of (state, symbol) slots with exactly one destination
    df["det_slot_frac"] = 1 - df["nondeterministic_pairs"] / (df["original_states"] * 2) - df["missing_slot_frac"]
    return df


def within_group_z(df, columns):
    # Each trait standardized inside its (class, n) group
    groups = df.groupby("group")
    z = pd.DataFrame(index=df.index)

    for column in columns:
        std = groups[column].transform("std").replace(0, np.nan)
        z[column] = ((df[column] - groups[column].transform("mean")) / std).fillna(0)

    return z


def mark_extreme(df, size_column, top_share):
    # Top `top_share` of each (class, n) group by DFA size
    return df.groupby("group")[size_column].rank(pct=True, method="average") > top_share


# ---------------------------------------------------------------- 1 and 3: significance and robustness

def compare_extremes(df, extreme, label):
    z = within_group_z(df, TRAIT_COLUMNS)
    rows = []

    for column in TRAIT_COLUMNS:
        in_extreme = z.loc[extreme, column]
        rest = z.loc[~extreme, column]
        rows.append({
            "definition": label,
            "trait": column,
            "z_difference": in_extreme.mean() - rest.mean(),
            "p_value": mannwhitneyu(in_extreme, rest).pvalue,
            "extreme_mean": df.loc[extreme, column].mean(),
            "rest_mean": df.loc[~extreme, column].mean(),
        })

    table = pd.DataFrame(rows).sort_values("z_difference", key=abs, ascending=False)
    return table, int(extreme.sum()), df.loc[extreme, "nfa_type"].value_counts().to_dict()


# ---------------------------------------------------------------- 2: predictive model

def fit_forest(df, extreme, seed=0):
    columns = TRAIT_COLUMNS + FOREST_EXTRA_COLUMNS
    features = within_group_z(df, columns).to_numpy()
    labels = extreme.to_numpy()
    forest = RandomForestClassifier(300, min_samples_leaf=10, n_jobs=-1, random_state=seed)

    probability = cross_val_predict(
        forest, features, labels, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba"
    )[:, 1]
    auc = roc_auc_score(labels, probability)

    # Importance is measured on the fitted forest as the AUC lost when one trait is shuffled
    forest.fit(features, labels)
    importance = permutation_importance(forest, features, labels, n_repeats=5, random_state=seed, scoring="roc_auc", n_jobs=-1)
    ranked = pd.Series(importance.importances_mean, index=columns).sort_values(ascending=False)
    return auc, ranked


def main():
    parser = argparse.ArgumentParser(description="Significance, predictive model and robustness checks for the feature search")
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_10K_DIR)
    args = parser.parse_args()

    out_dir = args.experiment_dir / "feature_search_extras"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = add_ratios(load_features(args.experiment_dir))
    df["log_subset"] = np.log2(df["subset_states"])
    print(f"{len(df)} random NFAs, {df['group'].nunique()} (class, n) groups")

    definitions = [
        ("minimized DFA, top 5 %", "log_dfa", 0.95),
        ("minimized DFA, top 1 %", "log_dfa", 0.99),
        ("subset DFA, top 5 %", "log_subset", 0.95),
    ]
    tables = []

    for label, column, top_share in definitions:
        extreme = mark_extreme(df, column, top_share)
        table, count, by_class = compare_extremes(df, extreme, label)
        tables.append(table)
        print(f"\n=== {label}: {count} extreme NFAs, by class {by_class}")
        print(table.head(8).round(4).to_string(index=False))

    pd.concat(tables).to_csv(out_dir / "z_differences_and_p_values.csv", index=False)

    extreme = mark_extreme(df, "log_dfa", 0.95)
    auc, ranked = fit_forest(df, extreme)
    ranked.rename("permutation_importance_auc_drop").round(4).to_csv(out_dir / "forest_importance.csv")
    pd.DataFrame([{"model": "random forest, 5-fold CV", "target": "top 5 % of minimized DFA size in group", "auc": auc}]).round(3).to_csv(
        out_dir / "forest_auc.csv", index=False
    )
    print(f"\nRandom forest on the within-group z-scored traits: cross-validated AUC = {auc:.3f}  (0.5 = chance)")
    print("Traits the forest relies on most (AUC lost when shuffled):")
    print(ranked.head(8).round(4).to_string())
    print(f"\nWrote CSVs to {out_dir}")


if __name__ == "__main__":
    main()
