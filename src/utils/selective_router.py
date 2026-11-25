from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn


class SelectiveSafetyRouter:
    """
    Router Human-in-the-Loop (HITL) que audita incertidumbre epistémica (MC-Dropout)
    y dependencia visual contrafactual antes de emitir una decisión autónoma.
    """
    def __init__(
        self,
        model: nn.Module,
        var_threshold: float = 0.015,
        visual_delta_threshold: float = 0.12,
        mc_passes: int = 5,
    ):
        self.model = model
        self.var_threshold = var_threshold
        self.visual_delta_threshold = visual_delta_threshold
        self.mc_passes = mc_passes

    def _enable_mc_dropout(self):
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @torch.no_grad()
    def route(self, img_map: torch.Tensor, text_emb: torch.Tensor) -> Dict[str, Any]:
        self.model.eval()
        out_real = self.model(img_map, text_emb)
        out_blind = self.model(torch.zeros_like(img_map), text_emb)

        prob_joint = float(torch.sigmoid(out_real["logit"]).item())
        prob_blind = float(torch.sigmoid(out_blind["logit"]).item())
        prob_text_aux = float(torch.sigmoid(out_real["logit_text"]).item())

        visual_dependency = abs(prob_joint - prob_blind)

        self._enable_mc_dropout()
        mc_probs = [
            float(torch.sigmoid(self.model(img_map, text_emb)["logit"]).item())
            for _ in range(self.mc_passes)
        ]
        epistemic_var = float(np.var(mc_probs))

        reason = None
        if epistemic_var > self.var_threshold:
            action = "FLAG_FOR_HUMAN_REVIEW"
            reason = "HIGH_EPISTEMIC_UNCERTAINTY"
        elif prob_joint >= 0.5 and visual_dependency < self.visual_delta_threshold:
            action = "FLAG_FOR_HUMAN_REVIEW"
            reason = "UNIMODAL_TEXT_SHORTCUT_DETECTED"
        else:
            action = "BLOCK" if prob_joint >= 0.5 else "ALLOW"

        return {
            "action": action,
            "reason": reason,
            "prob_joint": round(prob_joint, 4),
            "prob_text_only": round(prob_text_aux, 4),
            "visual_dependency_delta": round(visual_dependency, 4),
            "epistemic_variance": round(epistemic_var, 5),
        }
