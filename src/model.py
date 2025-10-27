import torch
import torch.nn as nn
from modules.transformer import TransformerEncoder
from cross_modal import MultiEncoder, PairwiseCrossModalFusionMean

class CUREEncoder(nn.Module):
    def __init__(
        self,
        clinical_dim: int,
        paraclinical_dim: int, 
        omics_dim: int,
        embed_dim: int = 128,
        transformer_cls=lambda: TransformerEncoder(embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1),
    ):
        super().__init__()
        self.multi_encoder = MultiEncoder(clinical_dim, paraclinical_dim, omics_dim, embed_dim)
        self.fusion = PairwiseCrossModalFusionMean(embed_dim, transformer_cls)

    @torch.no_grad()
    def encode(self, x_clin, x_para, x_omics):
        self.eval()
        zc, zp, zo = self.multi_encoder(x_clin, x_para, x_omics)
        z_fused = self.fusion(zc, zp, zo)
        return z_fused

    def forward(self, x_clin, x_para, x_omics):
        zc, zp, zo = self.multi_encoder(x_clin, x_para, x_omics)
        z_fused = self.fusion(zc, zp, zo)
        return z_fused