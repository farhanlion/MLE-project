#!/usr/bin/env python3
# coding: utf-8
"""
preprocessing_before_fit.py

Shared preprocessing only. Exports:
- preprocess_features_for_lr(train_df, other_dfs, categorical_cols)
    -> fits OHE+scaler on train_df, returns processed train_df and list of processed other_dfs,
       plus scaler object and lr_feature_column_names (ordered)
- preprocess_features_for_tree(df_list) -> simply fills missing flags and returns copies (trees use raw features)
- helper: add_missing_flag_and_fill(df) -> generic missing indicator + fillna(0)
"""

from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# NOTE: keep small & focused — no Spark code here.

def add_missing_flag_and_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["missing_any"] = df.isnull().any(axis=1).astype(int)
    df = df.fillna(0)
    return df

def preprocess_features_for_tree(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Minimal preprocessing for tree models (no scaling, preserve categorical raw values).
    Returns processed df list in same order.
    """
    out = []
    for df in dfs:
        out.append(add_missing_flag_and_fill(df))
    return out

def preprocess_features_for_lr(
    train_df: pd.DataFrame,
    other_dfs: List[pd.DataFrame],
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, List[pd.DataFrame], StandardScaler, List[str]]:
    """
    Fit one-hot encoding (via pandas.get_dummies on train_df), fit StandardScaler on numeric columns in train_df,
    and then apply the same columns+scaler to other_dfs so they have identical columns and scaling.

    Returns:
        train_proc: processed train DataFrame (numpy types)
        other_procs: list of processed other DataFrames in the same order as other_dfs
        scaler: fitted StandardScaler
        lr_cols: final ordered list of columns used for LR
    """
    # make copies
    train = train_df.copy()
    others = [d.copy() for d in other_dfs]

    # add missing flag + fill
    train = add_missing_flag_and_fill(train)
    others = [add_missing_flag_and_fill(d) for d in others]

    # one-hot encode categorical cols on train
    if categorical_cols:
        train_ohe = pd.get_dummies(train, columns=categorical_cols, drop_first=True, dtype=int)
    else:
        train_ohe = train.copy()

    # determine lr columns order
    lr_cols = train_ohe.columns.tolist()

    # ensure all other dfs have the same columns (create missing ones)
    def align_and_ohe(df):
        if categorical_cols:
            df_ohe = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
        else:
            df_ohe = df.copy()
        # add any missing columns with zeros
        for c in lr_cols:
            if c not in df_ohe.columns:
                df_ohe[c] = 0
        # keep only lr_cols order
        df_ohe = df_ohe[lr_cols]
        return df_ohe

    other_ohe = [align_and_ohe(d) for d in others]

    # detect numeric columns to scale (all columns with numeric dtype)
    numeric_cols = [c for c in lr_cols if pd.api.types.is_numeric_dtype(train_ohe[c])]
    scaler = StandardScaler()
    if len(numeric_cols) > 0:
        scaler.fit(train_ohe[numeric_cols])
        train_ohe[numeric_cols] = scaler.transform(train_ohe[numeric_cols])
        for df in other_ohe:
            df[numeric_cols] = scaler.transform(df[numeric_cols])

    # ensure dtypes are numeric (float64/int)
    train_ohe = train_ohe.astype(float)
    other_ohe = [d.astype(float) for d in other_ohe]

    return train_ohe, other_ohe, scaler, lr_cols
