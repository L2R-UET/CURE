import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as _tts
from data import load_clean_data
from cmhe_demo_utils import factual_evaluate
from metrics import cindex_from_predictions
from init import process_omics_embeddings, train_cross_modal, train_CURE

def run_experiment(
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    embed_dim: int = 128,
    horizons = (102, 177, 294)
):
    data = load_clean_data()

    omics_df = process_omics_embeddings(data['features'])

    clinical      = data['features']['clinical'].copy()
    paraclinical  = data['features']['paraclinical'].copy()
    demographics  = data['features']['demographics'].copy()
    treatment     = data['features']['treatment'].copy()
    outcomes      = data['outcomes'].copy()
    
    full = (omics_df
            .merge(clinical,     on="SAMPLE_ID", how="inner")
            .merge(paraclinical, on="SAMPLE_ID", how="inner")
            .merge(demographics, on="SAMPLE_ID", how="inner")
            .merge(treatment,    on="SAMPLE_ID", how="inner")
            .merge(outcomes,     on="SAMPLE_ID", how="inner"))

    omics_cols         = [c for c in omics_df.columns if c != "SAMPLE_ID"]
    clinical_cols      = [c for c in clinical.columns if c != "SAMPLE_ID"]
    paraclinical_cols  = [c for c in paraclinical.columns if c != "SAMPLE_ID"]
    demo_cols          = [c for c in demographics.columns if c != "SAMPLE_ID"]
    treatment_cols     = [c for c in treatment.columns if c != "SAMPLE_ID"]

    sids = full["SAMPLE_ID"].astype(str).tolist()
    train_ids, test_ids = _tts(sids, test_size=test_size, random_state=random_state)

    def select_ids(df: pd.DataFrame, ids: list):
        return df[df["SAMPLE_ID"].astype(str).isin(ids)].reset_index(drop=True)

    full_train = select_ids(full, train_ids)
    full_test  = select_ids(full, test_ids)

    y_cols = ["Overall Survival (Months)", "Overall Survival Status"]
    y_train = full_train[["SAMPLE_ID"] + y_cols].rename(
        columns={"Overall Survival (Months)": "time", "Overall Survival Status": "event"}
    )
    y_test  = full_test[["SAMPLE_ID"] + y_cols].rename(
        columns={"Overall Survival (Months)": "time", "Overall Survival Status": "event"}
    )

    fused_train, fused_test = train_cross_modal(
        full_train, full_test,
        clinical_cols=clinical_cols,
        paraclinical_cols=paraclinical_cols,
        omics_cols=omics_cols,
        embed_dim=embed_dim
    )
    fused_train_df = pd.DataFrame(fused_train.detach().cpu().numpy())
    fused_test_df  = pd.DataFrame(fused_test.detach().cpu().numpy())

    demo_train = full_train[demo_cols].reset_index(drop=True)
    demo_test  = full_test[demo_cols].reset_index(drop=True)

    if "Chemotherapy" in treatment_cols:
        chemo_train = full_train[["Chemotherapy"]].reset_index(drop=True)
        chemo_test  = full_test[["Chemotherapy"]].reset_index(drop=True)
    else:
        chemo_train = pd.DataFrame({"Chemotherapy": np.zeros(len(fused_train_df), dtype=float)})
        chemo_test  = pd.DataFrame({"Chemotherapy": np.zeros(len(fused_test_df), dtype=float)})

    final_train = pd.concat([fused_train_df, demo_train, chemo_train], axis=1)
    final_test  = pd.concat([fused_test_df,  demo_test,  chemo_test],  axis=1)

    if "Age at Diagnosis" in final_train.columns:
        scaler = StandardScaler()
        final_train["Age at Diagnosis"] = scaler.fit_transform(final_train[["Age at Diagnosis"]])
        final_test["Age at Diagnosis"]  = scaler.transform(final_test[["Age at Diagnosis"]])
        
    t_train = y_train["time"].values.astype(np.float32)
    e_train = y_train["event"].values.astype(np.float32)
    t_test  = y_test["time"].values.astype(np.float32)
    e_test  = y_test["event"].values.astype(np.float32)
    a_train = final_train["Chemotherapy"].values.astype(np.float32)
    a_test  = final_test["Chemotherapy"].values.astype(np.float32)
    
    model = train_CURE(
        final_train.values.astype(np.float32),
        t_train, e_train, a_train
    )
    
    predictions = model.predict_survival(final_test.values.astype(np.float32), a_test, t=list(horizons))

    CI1, CI3, CI5, IBS = factual_evaluate(
        (final_train.values, t_train, e_train, a_train),
        (final_test.values,  t_test,  e_test,  a_test),
        list(horizons), predictions
    )
    overall_c = float(cindex_from_predictions(t_test, e_test, list(horizons), predictions))

    metrics = {
        f"CI@{horizons[0]}": float(np.around(CI1, 4)),
        f"CI@{horizons[1]}": float(np.around(CI3, 4)),
        f"CI@{horizons[2]}": float(np.around(CI5, 4)),
        "C-index_overall": overall_c,
        "IBS": float(np.around(IBS, 4)),
    }
    return metrics, model, (final_train, final_test, y_train, y_test)
