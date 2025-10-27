import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np 
from numpy import random
import pandas as pd

class MoE_Layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.experts = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, output_dim) for _ in range(num_experts)
        ])

        self.gate = torch.nn.Linear(input_dim, num_experts)

    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        gate_logits = self.gate(x)

        aux_loss = self._compute_load_balancing_loss(gate_logits)

        topk_logits, topk_indices = torch.topk(gate_logits, self.top_k, dim=1)
 
        topk_weights = F.softmax(topk_logits, dim=1)  

        selected_experts = expert_outputs.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)))
        
        topk_weights = topk_weights.unsqueeze(-1)  
        output = torch.sum(topk_weights * selected_experts, dim=1)  
        
        return output, aux_loss
    def _compute_load_balancing_loss(self, gate_logits):
        """
        Tính auxiliary loss để cân bằng tải giữa các expert
        Sử dụng 2 thành phần:
        1. Importance loss: đảm bảo mỗi expert được sử dụng tương đương nhau
        2. Load loss: tránh quá tải expert
        """
        gates = F.softmax(gate_logits, dim=-1) 
        
        importance_per_expert = gates.mean(dim=0)  
        target_importance = torch.ones_like(importance_per_expert) / self.num_experts
        importance_loss = F.kl_div(
                importance_per_expert.log(),
                target_importance,
                reduction='sum'
            )
        
        return importance_loss
    
class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=(1024,), p=0.1):
        super().__init__()
        layers, last = [], d_in
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(p)]
            last = h
        layers += [nn.Linear(last, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MoE_AE(nn.Module):
    def __init__(self, d_in, bottleneck=128, num_experts=4, top_k=2,
                 enc_hidden=(1024,256), dec_hidden=(256,), pre_dim=512):
        super().__init__()
        self.pre_bottleneck = nn.Sequential(
            nn.Linear(d_in, pre_dim),
            nn.LayerNorm(pre_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        layers = []
        last = pre_dim
        for h in enc_hidden:
            layers += [
                nn.Linear(last, h), 
                nn.ReLU(), 
                nn.Dropout(0.1)
                ]
            last = h
        self.encoder = nn.Sequential(*layers)

        self.moe = MoE_Layer(input_dim=last, output_dim=bottleneck,
                             num_experts=num_experts, top_k=top_k)

        dec = []
        last = bottleneck
        for h in dec_hidden:
            dec += [
                nn.Linear(last, h), 
                nn.ReLU()
                ]
            last = h
        dec += [nn.Linear(last, d_in)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        h0 = self.pre_bottleneck(x)  
        h = self.encoder(h0)         
        z, aux = self.moe(h)       
        xhat = self.decoder(z)      
        return xhat, z, aux