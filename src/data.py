import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from modular_moe import MoE_AE 


def train_moe_modality(
    data_df: pd.DataFrame,
    input_dim: int,
    bottleneck: int = 256,
    num_experts: int = 4,
    top_k: int = 2,
    lambda_aux: float = 0.01,
    proj_hidden=(1024, 256),
    dec_hidden=(256,),
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 5e-4,
    wd: float = 1e-5,
    device: str | torch.device = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_np = data_df.values.astype(np.float32)

    X = torch.from_numpy(X_np)
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    moe = MoE_AE(
        d_in=input_dim,
        bottleneck=bottleneck,
        num_experts=num_experts,
        top_k=top_k,
        enc_hidden=proj_hidden,
        dec_hidden=dec_hidden,
    ).to(device)
    optimizer = torch.optim.Adam(moe.parameters(), lr=lr, weight_decay=wd)

    moe.train()
    for epoch in range(epochs):
        total = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            xhat, z, aux = moe(xb)                
            loss = F.mse_loss(xhat, xb) + lambda_aux * aux       

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * xb.size(0)
        if epoch % 10 == 0 or epoch == 1:
            rmse = (total / len(ds)) ** 0.5
            print(f"[{epoch:03d}] loss={loss.item():.6f} | RMSE={rmse:.6f}")

    moe.eval()
    with torch.no_grad():
        embs = []
        for (xb,) in DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False):
            xb = xb.to(device)
            _, z, _ = moe(xb)
            embs.append(z.detach().cpu().numpy())
        embeddings = np.concatenate(embs, axis=0)  # (n_samples, bottleneck)

    return embeddings 

def load_omics_embeddings(
    cna_path="../datasets/metabric/raw_cna.csv",
    meth_path="../datasets/metabric/raw_meth.csv",
    mrna_path="../datasets/metabric/raw_mrna.csv",
    mut_path="../datasets/metabric/raw_mut.csv",
):
    cna_df  = pd.read_csv(cna_path)
    meth_df = pd.read_csv(meth_path)
    mrna_df = pd.read_csv(mrna_path)
    mut_df  = pd.read_csv(mut_path)

    # Tách features (bỏ SAMPLE_ID nếu có)
    def _features(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=["SAMPLE_ID"]) if "SAMPLE_ID" in df.columns else df

    cna_X  = _features(cna_df)
    meth_X = _features(meth_df)
    mrna_X = _features(mrna_df)
    mut_X  = _features(mut_df)

    cna_X = cna_X.fillna(cna_X.median())
    meth_X = meth_X.fillna(meth_X.median())
    mrna_X = mrna_X.fillna(mrna_X.median())
    mut_X = mut_X.fillna(0)

    cna_emb  = train_moe_modality(cna_X,  input_dim=cna_X.shape[1],  bottleneck=256, epochs=150)
    meth_emb = train_moe_modality(meth_X, input_dim=meth_X.shape[1], bottleneck=256, epochs=150)
    mrna_emb = train_moe_modality(mrna_X, input_dim=mrna_X.shape[1], bottleneck=256,  epochs=150)
    mut_emb  = train_moe_modality(mut_X,  input_dim=mut_X.shape[1],  bottleneck=64,  epochs=150)

    cna_emb_df  = pd.DataFrame(cna_emb,  columns=[f"cna_z{i}"  for i in range(cna_emb.shape[1])])
    meth_emb_df = pd.DataFrame(meth_emb, columns=[f"meth_z{i}" for i in range(meth_emb.shape[1])])
    mrna_emb_df = pd.DataFrame(mrna_emb, columns=[f"mrna_z{i}" for i in range(mrna_emb.shape[1])])
    mut_emb_df  = pd.DataFrame(mut_emb,  columns=[f"mut_z{i}"  for i in range(mut_emb.shape[1])])

    if "SAMPLE_ID" in cna_df.columns:
        cna_emb_df.index  = cna_df["SAMPLE_ID"]
    if "SAMPLE_ID" in meth_df.columns:
        meth_emb_df.index = meth_df["SAMPLE_ID"]
    if "SAMPLE_ID" in mrna_df.columns:
        mrna_emb_df.index = mrna_df["SAMPLE_ID"]
    if "SAMPLE_ID" in mut_df.columns:
        mut_emb_df.index  = mut_df["SAMPLE_ID"]

    return {
        "cna_emb": cna_emb_df,
        "methylation_emb": meth_emb_df,
        "mrna_emb": mrna_emb_df,
        "mutation_emb": mut_emb_df,
    }


def load_clinical_data():
    x_clinical      = pd.read_csv("../datasets/metabric/preprocessed_metabric_clinical.csv")
    x_paraclinical  = pd.read_csv("../datasets/metabric/preprocessed_metabric_paraclinical.csv")
    x_treatment     = pd.read_csv("../datasets/metabric/preprocessed_metabric_treatment.csv")
    x_demographics  = pd.read_csv("../datasets/metabric/preprocessed_metabric_demographics.csv")
    y               = pd.read_csv("../datasets/metabric/preprocessed_metabric_output.csv")
    return {
        "clinical": x_clinical,
        "paraclinical": x_paraclinical,
        "treatment": x_treatment,
        "demographics": x_demographics,
        "outcomes": y,
    }


def load_data():
    omics_emb = load_omics_embeddings()
    clinical  = load_clinical_data()

    features = {
        "cna_emb": omics_emb["cna_emb"],
        "methylation_emb": omics_emb["methylation_emb"],
        "mrna_emb": omics_emb["mrna_emb"],
        "mutation_emb": omics_emb["mutation_emb"],
        "clinical": clinical["clinical"],
        "paraclinical": clinical["paraclinical"],
        "treatment": clinical["treatment"],
        "demographics": clinical["demographics"],
    }
    return {"features": features, "outcomes": clinical["outcomes"]}

def load_clean_data():
    x_clinical      = pd.read_csv("../datasets/metabric/preprocessed_metabric_clinical.csv")
    x_paraclinical  = pd.read_csv("../datasets/metabric/preprocessed_metabric_paraclinical.csv")
    x_treatment     = pd.read_csv("../datasets/metabric/preprocessed_metabric_treatment.csv")
    x_demographics  = pd.read_csv("../datasets/metabric/preprocessed_metabric_demographics.csv")
    y               = pd.read_csv("../datasets/metabric/preprocessed_metabric_output.csv")
    cna_moe = pd.read_csv("../datasets/metabric/moe_cna.csv")
    meth_moe = pd.read_csv("../datasets/metabric/moe_meth.csv")
    mrna_moe = pd.read_csv("../datasets/metabric/moe_mrna.csv")
    mut_moe = pd.read_csv("../datasets/metabric/moe_mut.csv")
    features = {
        "cna_emb": cna_moe,
        "methylation_emb": meth_moe,
        "mrna_emb": mrna_moe,
        "mutation_emb": mut_moe,
        "clinical": x_clinical,
        "paraclinical": x_paraclinical,
        "treatment": x_treatment,
        "demographics": x_demographics,
    }
    return {"features": features, "outcomes": y}