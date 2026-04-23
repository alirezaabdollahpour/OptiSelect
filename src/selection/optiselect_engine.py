"""
OptiSelect Engine: Online in-batch data selection with optimizer-aware scoring.

Implements the core OPUS-style pipeline:
  1. Forward-backward on 2B candidates -> capture Ghost factors (a(z), b(z))
  2. Compute validation gradient factors (averaged over proxy)
  3. Score candidates using optimizer-specific frozen-state operator O_t
  4. Greedy Boltzmann selection with redundancy penalty (Paper Eq. 4)
  5. Train on selected B samples using standard optimizer step

Numerical robustness:
  - All scoring outputs sanitized (NaN/Inf -> sentinel values)
  - Safe Boltzmann sampling via max-subtraction softmax
  - Fallback to uniform sampling if all probs underflow to 0
  - Final multinomial wrapped in try/except with argmax fallback
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from selection.influence_scoring import (
    InfluenceConfig,
    compute_adam_family_scores,
    compute_lion_scores,
    compute_sophia_scores,
    compute_muon_scores,
    compute_soap_scores,
    compute_galore_scores,
    compute_cadamw_scores,
    extract_optimizer_preconditioner,
)


class OptiSelectEngine:
    """Engine for optimizer-aware online in-batch data selection."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        opt_name: str,
        config: InfluenceConfig,
        device: str = "cuda",
    ):
        self.model = model
        self.optimizer = optimizer
        self.opt_name = opt_name
        self.config = config
        self.device = device

        self._val_layer_activations = {}
        self._val_layer_backprops = {}

        self._hooks = []
        self._layer_activations = {}
        self._layer_backprops = {}
        self._target_layers = {}

        self._is_capturing = False

        self.selection_stats = {
            "entropy": [],
            "mean_score": [],
            "std_score": [],
        }

        self._register_ghost_hooks()

    # ------------------------------------------------------------------
    # Ghost factor capture (hooks)
    # ------------------------------------------------------------------

    def _register_ghost_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self._target_layers[name] = module
                fwd_hook = module.register_forward_hook(self._make_forward_hook(name))
                bwd_hook = module.register_full_backward_hook(self._make_backward_hook(name))
                self._hooks.extend([fwd_hook, bwd_hook])

    def _make_forward_hook(self, layer_name: str):
        def hook(module, input, output):
            if not self._is_capturing:
                return
            self._layer_activations[layer_name] = input[0].detach()
        return hook

    def _make_backward_hook(self, layer_name: str):
        def hook(module, grad_input, grad_output):
            if not self._is_capturing:
                return
            self._layer_backprops[layer_name] = grad_output[0].detach()
        return hook

    def start_capture(self):
        self._is_capturing = True
        self._layer_activations.clear()
        self._layer_backprops.clear()

    def stop_capture(self):
        self._is_capturing = False

    def detach(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._layer_activations.clear()
        self._layer_backprops.clear()

    # ------------------------------------------------------------------
    # Validation gradient proxy (Paper Section 4.2)
    # ------------------------------------------------------------------

    def compute_validation_gradient_factors(
        self,
        model,
        val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        type_ctx,
    ):
        """Aggregate Ghost factors across proxy, average, cache."""
        model.eval()

        accum_a: Dict[str, torch.Tensor] = {}
        accum_b: Dict[str, torch.Tensor] = {}
        total_tokens = 0

        for val_x, val_y in val_batches:
            self.start_capture()
            model.zero_grad()
            with type_ctx:
                outputs = model(val_x, targets=val_y)
            outputs["loss"].backward()
            self.stop_capture()

            for name in self._layer_activations:
                a = self._layer_activations[name].detach()
                b = self._layer_backprops[name].detach()
                a_sum = a.sum(dim=tuple(range(a.dim() - 1)))
                b_sum = b.sum(dim=tuple(range(b.dim() - 1)))
                if name not in accum_a:
                    accum_a[name] = a_sum
                    accum_b[name] = b_sum
                else:
                    accum_a[name] += a_sum
                    accum_b[name] += b_sum
            total_tokens += val_x.size(0) * val_x.size(1)

        self._val_layer_activations = {
            name: (accum_a[name] / total_tokens).unsqueeze(0).unsqueeze(0)
            for name in accum_a
        }
        self._val_layer_backprops = {
            name: (accum_b[name] / total_tokens).unsqueeze(0).unsqueeze(0)
            for name in accum_b
        }

        model.zero_grad()
        model.train()

    # ------------------------------------------------------------------
    # MARS raw-gradient variance tracking (Paper Remark 4)
    # ------------------------------------------------------------------

    def update_mars_scoring_v(self, beta2: float = 0.999):
        if self.opt_name != "mars":
            return
        for name, module in self._target_layers.items():
            if module.weight.grad is None:
                continue
            g_sq = module.weight.grad.detach() ** 2
            if not hasattr(module.weight, '_scoring_v'):
                module.weight._scoring_v = torch.zeros_like(module.weight)
            module.weight._scoring_v.mul_(beta2).add_(g_sq, alpha=(1 - beta2))

    # ------------------------------------------------------------------
    # ADOPT lagged variance caching (Paper Section 4.3.1)
    # ------------------------------------------------------------------

    def cache_adopt_prev_v(self):
        if self.opt_name != "adopt":
            return
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state.get(p, {})
                if "exp_avg_sq" in state:
                    state["_prev_exp_avg_sq"] = state["exp_avg_sq"].clone()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        candidate_activations: Dict[str, torch.Tensor],
        candidate_backprops: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute optimizer-aware influence scores with per-layer sanitization."""
        total_scores: Optional[torch.Tensor] = None

        for layer_name in candidate_activations:
            if layer_name not in self._val_layer_activations:
                continue

            cand_a = candidate_activations[layer_name]
            cand_b = candidate_backprops[layer_name]
            val_a = self._val_layer_activations[layer_name]
            val_b = self._val_layer_backprops[layer_name]

            layer = self._target_layers[layer_name]
            precond_info = extract_optimizer_preconditioner(
                self.optimizer, layer.weight, self.opt_name
            )

            if precond_info["type"] == "adam_family":
                layer_scores = compute_adam_family_scores(
                    cand_a, cand_b, val_a, val_b,
                    precond_info["preconditioner"].reshape(layer.weight.shape),
                )
            elif precond_info["type"] == "lion":
                layer_scores = compute_lion_scores(
                    cand_a, cand_b, val_a, val_b,
                    precond_info["sign_ut"].reshape(layer.weight.shape),
                )
            elif precond_info["type"] == "sophia":
                layer_scores = compute_sophia_scores(
                    cand_a, cand_b, val_a, val_b,
                    precond_info["hessian_diag"].reshape(layer.weight.shape),
                )
            elif precond_info["type"] == "muon":
                layer_scores = compute_muon_scores(
                    cand_a, cand_b, val_a, val_b,
                    precond_info["momentum_matrix"],
                )
            elif precond_info["type"] == "soap":
                layer_scores = compute_soap_scores(
                    cand_a, cand_b, val_a, val_b,
                    precond_info["U_L"], precond_info["U_R"],
                    precond_info["v_rotated"].reshape(layer.weight.shape),
                )
            elif precond_info["type"] == "galore":
                layer_scores = compute_galore_scores(
                    cand_a, cand_b, val_a, val_b,
                    precond_info["Pi_L"], precond_info["Pi_R"],
                    precond_info["v_projected"],
                )
            elif precond_info["type"] == "cadamw":
                layer_scores = compute_cadamw_scores(
                    cand_a, cand_b, val_a, val_b,
                    precond_info["preconditioner"].reshape(layer.weight.shape),
                    precond_info["mask"].reshape(layer.weight.shape),
                    precond_info["scale"],
                )
            else:
                # SGD fallback: raw gradient inner product
                layer_scores = compute_adam_family_scores(
                    cand_a, cand_b, val_a, val_b,
                    torch.ones_like(layer.weight),
                )

            # Sanitize per-layer before aggregation
            layer_scores = torch.nan_to_num(
                layer_scores, nan=0.0, posinf=1e9, neginf=-1e9
            )

            if total_scores is None:
                total_scores = layer_scores
            else:
                total_scores = total_scores + layer_scores

        if total_scores is None:
            raise RuntimeError("No layers found for scoring")

        # Final sanitization
        total_scores = torch.nan_to_num(
            total_scores, nan=0.0, posinf=1e9, neginf=-1e9
        )
        return total_scores

    # ------------------------------------------------------------------
    # Safe Boltzmann sampling
    # ------------------------------------------------------------------

    def _safe_boltzmann_sample(
        self,
        scores: torch.Tensor,
        available_mask: torch.Tensor,
        temperature: float,
    ) -> int:
        """
        Numerically robust single-sample from Boltzmann distribution.

        Guards against:
          - NaN / Inf in input scores
          - Overflow in exp(scores / temperature)
          - Underflow of all probabilities to 0
          - Negative probs from fp rounding
          - CUDA multinomial errors
        """
        # Sanitize input
        scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        scores = scores.masked_fill(~available_mask, float('-inf'))

        n_avail = int(available_mask.sum().item())
        if n_avail <= 0:
            raise RuntimeError("No candidates available for selection")
        if n_avail == 1:
            # Only one candidate left; pick it deterministically
            return int(torch.argmax(available_mask.long()).item())

        available_scores = scores.masked_select(available_mask)
        max_score = available_scores.max()

        if not torch.isfinite(max_score):
            # All -inf (shouldn't happen) -> uniform fallback
            probs = available_mask.float() / n_avail
            try:
                return int(torch.multinomial(probs, 1).item())
            except RuntimeError:
                return int(torch.argmax(available_mask.long()).item())

        # Max-subtraction softmax with clamping
        scores_shifted = (scores - max_score) / max(temperature, 1e-6)
        scores_shifted = scores_shifted.clamp(min=-50.0, max=50.0)

        probs = torch.exp(scores_shifted)
        probs = probs.masked_fill(~available_mask, 0.0)

        # Guard against non-finite or negative values
        probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
        probs = probs.clamp(min=0.0)

        total = probs.sum()
        if not torch.isfinite(total) or total.item() < 1e-20:
            # Fallback: uniform over available
            probs = available_mask.float()
            total = probs.sum()

        probs = probs / total

        # Final sanity check
        if torch.isnan(probs).any() or (probs < 0).any() or probs.sum().item() < 1e-10:
            probs = available_mask.float() / max(float(n_avail), 1.0)

        try:
            idx = int(torch.multinomial(probs, 1).item())
        except RuntimeError:
            # Last resort: deterministic argmax of scored candidates
            masked_scores = scores.masked_fill(~available_mask, float('-inf'))
            idx = int(torch.argmax(masked_scores).item())

        return idx

    # ------------------------------------------------------------------
    # Selection with redundancy penalty (Paper Eq. 4)
    # ------------------------------------------------------------------

    def select_batch_with_redundancy(
        self,
        candidate_activations: Dict[str, torch.Tensor],
        candidate_backprops: Dict[str, torch.Tensor],
        alignment_scores: torch.Tensor,
        n_select: int,
        eta: float = 1e-3,
        lambda_r: float = 1.0,
    ) -> torch.Tensor:
        """
        Greedy Boltzmann selection with redundancy penalty.

        Paper Eq. 4: U(z,t) = η ⟨u_z, g_V⟩ - η² λ_r ⟨u_z, G_t⟩
        """
        B_cand = alignment_scores.size(0)
        device = alignment_scores.device
        temperature = self.config.temperature

        alignment_scores = torch.nan_to_num(
            alignment_scores, nan=0.0, posinf=1e9, neginf=-1e9
        )

        selected: List[int] = []
        available_mask = torch.ones(B_cand, dtype=torch.bool, device=device)

        G_t = {
            name: {"a_sum": None, "b_sum": None, "count": 0}
            for name in candidate_activations.keys()
        }

        for round_idx in range(n_select):
            redundancy = torch.zeros(B_cand, device=device, dtype=torch.float32)

            if round_idx > 0:
                for name in candidate_activations:
                    cand_a = candidate_activations[name].to(torch.float32)
                    cand_b = candidate_backprops[name].to(torch.float32)
                    a_sum = G_t[name]["a_sum"]
                    b_sum = G_t[name]["b_sum"]
                    cnt = G_t[name]["count"]
                    if cnt == 0 or a_sum is None:
                        continue
                    a_mean = a_sum / cnt
                    b_mean = b_sum / cnt
                    a_dot = (cand_a * a_mean.unsqueeze(0)).sum(dim=(1, 2))
                    b_dot = (cand_b * b_mean.unsqueeze(0)).sum(dim=(1, 2))
                    redundancy = redundancy + a_dot * b_dot

            redundancy = torch.nan_to_num(
                redundancy, nan=0.0, posinf=1e9, neginf=-1e9
            )

            scores = eta * alignment_scores - (eta ** 2) * lambda_r * redundancy

            idx = self._safe_boltzmann_sample(scores, available_mask, temperature)

            selected.append(idx)
            available_mask[idx] = False

            for name in candidate_activations:
                a_z = candidate_activations[name][idx].to(torch.float32)
                b_z = candidate_backprops[name][idx].to(torch.float32)
                if G_t[name]["a_sum"] is None:
                    G_t[name]["a_sum"] = a_z.clone()
                    G_t[name]["b_sum"] = b_z.clone()
                else:
                    G_t[name]["a_sum"] += a_z
                    G_t[name]["b_sum"] += b_z
                G_t[name]["count"] += 1

        # Log entropy safely
        with torch.no_grad():
            temp_scores = torch.nan_to_num(
                alignment_scores, nan=0.0, posinf=1e9, neginf=-1e9
            )
            max_s = temp_scores.max()
            if torch.isfinite(max_s):
                scaled = ((temp_scores - max_s) / max(temperature, 1e-6)).clamp(
                    min=-50.0, max=50.0
                )
                probs0 = torch.exp(scaled)
                tot = probs0.sum().clamp(min=1e-20)
                probs0 = probs0 / tot
                entropy = -(probs0 * torch.log(probs0 + 1e-10)).sum().item()
                if not math.isfinite(entropy):
                    entropy = 0.0
            else:
                entropy = 0.0

            self.selection_stats["entropy"].append(entropy)
            self.selection_stats["mean_score"].append(
                float(alignment_scores.mean().item())
            )
            self.selection_stats["std_score"].append(
                float(alignment_scores.std().item())
                if alignment_scores.numel() > 1 else 0.0
            )

        return torch.tensor(selected, device=device, dtype=torch.long)

    def get_selection_summary(self) -> Dict:
        if not self.selection_stats["entropy"]:
            return {}
        n = len(self.selection_stats["entropy"])
        return {
            "mean_entropy": sum(self.selection_stats["entropy"]) / n,
            "mean_score": sum(self.selection_stats["mean_score"]) / n,
            "mean_std_score": sum(self.selection_stats["std_score"]) / n,
        }