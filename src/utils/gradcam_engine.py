import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeterministicGradCAM:
    """Extractor espacial mediante Hooks de PyTorch con limpieza determinista."""
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = []

    def _register_hooks(self):
        self.handles.append(
            self.target_layer.register_forward_hook(lambda m, i, o: setattr(self, "activations", o.detach()))
        )
        self.handles.append(
            self.target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "gradients", go[0].detach()))
        )

    def _remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def compute_visual_evidence(
        self, img_input: torch.Tensor, text_emb: torch.Tensor
    ) -> tuple[float, float, np.ndarray]:
        self.model.eval()
        self._register_hooks()
        try:
            img_req = img_input.clone().detach().requires_grad_(True)
            self.model.zero_grad(set_to_none=True)

            out = self.model(img_req, text_emb)
            logit = out["logit"].squeeze() if isinstance(out, dict) else out.squeeze()
            logit.backward()

            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = F.relu(torch.sum(weights * self.activations, dim=1).squeeze(0))
            cam_norm = cam / cam.max() if cam.max() > 0 else cam
            visual_mass = float((cam_norm > 0.5).float().mean().item())

            return float(logit.item()), visual_mass, cam_norm.cpu().numpy()
        finally:
            self._remove_hooks()
            self.activations = None
            self.gradients = None
