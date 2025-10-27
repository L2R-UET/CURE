import torch
import numpy as np
import pandas as pd
from model import CUREEncoder
from modules.transformer import TransformerEncoder
from auton_survival.models.cmhe import DeepCoxMixturesHeterogenousEffects

def process_omics_embeddings(features: dict) -> pd.DataFrame:
    parts = []
    for k in ("cna_emb", "methylation_emb", "mrna_emb", "mutation_emb"):
        df = features.get(k)
        if df is None or df.empty:
            continue
        df = df.copy()
        if "SAMPLE_ID" in df.columns:
            cols = [c for c in df.columns if c != "SAMPLE_ID"]
            df.columns = ["SAMPLE_ID"] + [f"{k}_{c}" for c in cols]
        else:
            df = df.rename_axis("SAMPLE_ID").reset_index()
            cols = [c for c in df.columns if c != "SAMPLE_ID"]
            df.columns = ["SAMPLE_ID"] + [f"{k}_{c}" for c in cols]
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["SAMPLE_ID"])
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="SAMPLE_ID", how="inner")
    return out


def train_cross_modal(
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    *,
    clinical_cols,
    paraclinical_cols,
    omics_cols,
    embed_dim: int = 128
):
    def df_to_tensor(df: pd.DataFrame, cols):
        block = df.loc[:, cols].copy()
        for col in ("SAMPLE_ID", "index"):
            if col in block.columns:
                block.drop(columns=[col], inplace=True, errors="ignore")
        num = block.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            raise RuntimeError("No numeric columns in provided column list.")
        return torch.tensor(num.values.astype(np.float32))
    
    xC_tr = df_to_tensor(features_train, clinical_cols)
    xP_tr = df_to_tensor(features_train, paraclinical_cols)
    xO_tr = df_to_tensor(features_train, omics_cols)

    xC_te = df_to_tensor(features_test, clinical_cols)
    xP_te = df_to_tensor(features_test, paraclinical_cols)
    xO_te = df_to_tensor(features_test, omics_cols)

    clinical_dim     = xC_tr.shape[1]
    paraclinical_dim = xP_tr.shape[1]
    omics_dim        = xO_tr.shape[1]

    encoder = CUREEncoder(
        clinical_dim=clinical_dim,
        paraclinical_dim=paraclinical_dim,
        omics_dim=omics_dim,
        embed_dim=embed_dim,
        transformer_cls=lambda: TransformerEncoder(
            embed_dim=embed_dim, num_heads=4, layers=2,
            attn_dropout=0.1, relu_dropout=0.1,
            res_dropout=0.1, embed_dropout=0.1, attn_mask=False
        )
    )

    fused_train = encoder.encode(xC_tr, xP_tr, xO_tr)
    fused_test  = encoder.encode(xC_te, xP_te, xO_te)
    return fused_train, fused_test

def train_CURE(x_train, t_train, e_train, a_train, k=2, g=3):
    
    config = dict(
        k=k,
        g=g,
        layers=[50, 50],
        random_seed=42,
        iters=20,
        learning_rate=0.01,
        batch_size=100,
        vsize=0.15,
        patience=3,
        optimizer="Adam"
    )

    print(f"Training CURE with k={k} phenotypes, g={g} treatment effects")
    
    model = DeepCoxMixturesHeterogenousEffects(
        random_seed=config["random_seed"],
        k=config["k"],
        g=config["g"],
        layers=config["layers"]
    )

    model = model.fit(
        x_train, t_train, e_train, a_train,
        vsize=config["vsize"],
        val_data=None,
        iters=config["iters"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        optimizer=config["optimizer"],
        patience=config["patience"]
    )

    return model