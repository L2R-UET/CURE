import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim) 
        )

    def forward(self, x):
        return self.net(x)

class MultiEncoder(nn.Module):
        def __init__(self, clinical_dim, paraclinical_dim, omics_dim, embed_dim):
            super().__init__()
            self.clinical_encoder = MLPEncoder(clinical_dim, embed_dim)
            self.paraclinical_encoder = MLPEncoder(paraclinical_dim, embed_dim)
            self.omics_encoder = MLPEncoder(omics_dim, embed_dim)

        def forward(self, x_clinical, x_paraclinical, x_omics):
            z_clin = self.clinical_encoder(x_clinical)
            z_para = self.paraclinical_encoder(x_paraclinical)
            z_omics = self.omics_encoder(x_omics)
            return z_clin, z_para, z_omics

class PairwiseCrossModalFusionMean(nn.Module):
    def __init__(self, embed_dim, transformer_cls):
        super().__init__()

        # Clinical attends to P and O
        self.C_attn_P = transformer_cls()
        self.C_attn_O = transformer_cls()

        # Paraclinical attends to C and O
        self.P_attn_C = transformer_cls()
        self.P_attn_O = transformer_cls()

        # Omics attends to C and P
        self.O_attn_C = transformer_cls()
        self.O_attn_P = transformer_cls()

    def forward(self, C, P, O):
        C_seq, P_seq, O_seq = C.unsqueeze(0), P.unsqueeze(0), O.unsqueeze(0)
        print(f"C_seq shape: {C_seq.shape}")

        C_from_P = self.C_attn_P(x_in=C_seq, x_in_k=P_seq, x_in_v=P_seq)[0]  # [B, D]
        C_from_O = self.C_attn_O(x_in=C_seq, x_in_k=O_seq, x_in_v=O_seq)[0]  # [B, D]

        P_from_C = self.P_attn_C(x_in=P_seq, x_in_k=C_seq, x_in_v=C_seq)[0]
        P_from_O = self.P_attn_O(x_in=P_seq, x_in_k=O_seq, x_in_v=O_seq)[0]

        O_from_C = self.O_attn_C(x_in=O_seq, x_in_k=C_seq, x_in_v=C_seq)[0]
        O_from_P = self.O_attn_P(x_in=O_seq, x_in_k=P_seq, x_in_v=P_seq)[0]

        print(f"O_from_C shape: {O_from_C.shape}")

        fused = torch.mean(
            torch.stack([C_from_P, C_from_O, P_from_C, P_from_O, O_from_C, O_from_P], dim=0),
            dim=0
        ) 

        return fused