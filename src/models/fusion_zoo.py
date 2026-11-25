from typing import Dict
import torch
import torch.nn as nn


class CoordConv2d(nn.Module):
    """Inyecta coordenadas espaciales (X, Y) normalizadas en [-1, 1] antes de proyectar."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        y_grid = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        x_grid = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return self.conv(torch.cat([x, x_grid, y_grid], dim=1))


class MultimodalFusionZoo(nn.Module):
    """
    Arquitectura experimental unificada para evaluar:
    - Baselines unimodales: 'text_only', 'image_only'
    - Fusión desalineada (DistilBERT + ResNet-50): 'concat', 'hadamard', 'cross_attn'
    - Fusión alineada contrastivamente (CLIP ViT-B/32): 'clip_concat', 'clip_hadamard'
    """
    def __init__(
        self,
        arch: str = "cross_attn",
        d_model: int = 256,
        resnet_channels: int = 2048,
        bert_dim: int = 768,
        clip_dim: int = 512,
        nhead: int = 8,
        dropout: float = 0.3,
        modality_dropout: float = 0.25,
    ):
        super().__init__()
        self.arch = arch
        self.is_clip = arch.startswith("clip_")
        self.modality_dropout = modality_dropout

        if self.is_clip:
            self.proj_clip_img = nn.Sequential(nn.Linear(clip_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
            self.proj_clip_txt = nn.Sequential(nn.Linear(clip_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        else:
            self.proj_img = nn.Sequential(
                CoordConv2d(resnet_channels, d_model),
                nn.BatchNorm2d(d_model),
                nn.GELU(),
            )
            self.proj_text = nn.Sequential(
                nn.Linear(bert_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )

        # Cabezales auxiliares unimodales para auditar atajos en inferencia
        self.aux_text = nn.Linear(d_model, 1)
        self.aux_img = nn.Linear(d_model, 1)

        if arch == "cross_attn":
            self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            in_dim = d_model * 2
        elif arch in ["concat", "clip_concat"]:
            in_dim = d_model * 2
        elif arch in ["hadamard", "clip_hadamard"]:
            in_dim = d_model * 3  # [v, t, v * t]
        else:
            in_dim = d_model

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        resnet_map: torch.Tensor,
        bert_emb: torch.Tensor,
        clip_img: torch.Tensor | None = None,
        clip_txt: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        b = bert_emb.size(0)

        # Modality Dropout estocástico sobre el texto durante entrenamiento
        if self.training and self.modality_dropout > 0.0 and self.arch not in ["text_only", "image_only"]:
            keep_mask = torch.bernoulli(torch.full((b, 1), 1.0 - self.modality_dropout, device=bert_emb.device))
            bert_emb = bert_emb * keep_mask
            if clip_txt is not None:
                clip_txt = clip_txt * keep_mask

        if self.is_clip:
            v_global = self.proj_clip_img(clip_img.float())
            t_vec = self.proj_clip_txt(clip_txt.float())
            v_spatial = None
        else:
            v_spatial = self.proj_img(resnet_map.float())
            v_global = v_spatial.mean(dim=(2, 3))
            t_vec = self.proj_text(bert_emb.float())

        logit_t = self.aux_text(t_vec).squeeze(1)
        logit_v = self.aux_img(v_global).squeeze(1)

        if self.arch == "text_only":
            joint = logit_t
        elif self.arch == "image_only":
            joint = logit_v
        elif self.arch in ["concat", "clip_concat"]:
            joint = self.classifier(torch.cat([v_global, t_vec], dim=1)).squeeze(1)
        elif self.arch in ["hadamard", "clip_hadamard"]:
            joint = self.classifier(torch.cat([v_global, t_vec, v_global * t_vec], dim=1)).squeeze(1)
        elif self.arch == "cross_attn":
            tokens = v_spatial.flatten(2).transpose(1, 2)  # [B, 49, d]
            query = t_vec.unsqueeze(1)                     # [B, 1, d]
            attn_out, _ = self.attn(query, tokens, tokens)
            ctx = self.norm(query + attn_out).squeeze(1)
            joint = self.classifier(torch.cat([ctx, t_vec], dim=1)).squeeze(1)
        else:
            raise ValueError(f"Arquitectura no soportada: {self.arch}")

        return {
            "logit": joint,
            "logit_text": logit_t,
            "logit_image": logit_v,
        }