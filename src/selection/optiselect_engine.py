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

import hashlib
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from selection.influence_scoring import (
    InfluenceConfig,
    OuterProductCountSketch,
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
        self._proxy_sketches = {}
        self._countsketches = {}
        self._last_candidate_sketches = None

        self._hooks = []
        self._layer_activations = {}
        self._layer_backprops = {}
        self._target_layers = {}

        self._is_capturing = False

        self.selection_stats = {
            "entropy": [],
            "mean_score": [],
            "std_score": [],
            "utility_std": [],
            "eta_score_std_over_temperature": [],
            "effective_entropy_over_log_candidates": [],
            "selected_score_mean": [],
            "candidate_score_mean": [],
            "selected_score_percentile": [],
        }

        self._register_ghost_hooks()

    # ------------------------------------------------------------------
    # Ghost factor capture (hooks)
    # ------------------------------------------------------------------

    def _register_ghost_hooks(self):
        # Skip vocabulary-sized readout layers: their per-sample backprops are
        # O(B·seq·vocab), dominating peak memory, while the selection signal
        # lives in transformer-body layers (Paper Section 4.3).
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if "lm_head" in name or module.out_features > 32768:
                    continue
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
        self._proxy_sketches.clear()
        self._last_candidate_sketches = None

    def reduce_ghost_factor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert captured token-level Ghost factors to legacy per-sample factors.

        This helper is kept for callers that intentionally use a low-memory
        approximation. The CountSketch path below does not call it, because the
        true sequence-level gradient is sum_t b_t(z) ⊗ a_t(z), not
        outer(sum_t b_t(z), sum_t a_t(z)).
        """
        factor = tensor.detach().to(torch.float32)
        if factor.dim() > 2:
            factor = factor.sum(dim=tuple(range(1, factor.dim() - 1)))
        return factor

    def _sketch_key(self, layer_name: str, d_in: int, d_out: int):
        return layer_name, int(d_in), int(d_out)

    def _sketch_seed(self, layer_name: str, d_in: int, d_out: int) -> int:
        payload = f"{layer_name}:{d_in}:{d_out}:{self.config.sketch_dim}".encode()
        return int(hashlib.sha256(payload).hexdigest()[:8], 16)

    def _get_countsketch(
        self,
        layer_name: str,
        d_in: int,
        d_out: int,
        device,
    ) -> OuterProductCountSketch:
        key = self._sketch_key(layer_name, d_in, d_out)
        sketcher = self._countsketches.get(key)
        if sketcher is None:
            sketcher = OuterProductCountSketch(
                d_in=d_in,
                d_out=d_out,
                m=int(self.config.sketch_dim),
                seed=self._sketch_seed(layer_name, d_in, d_out),
                row_block=int(getattr(self.config, "countsketch_row_block", 32)),
            )
            self._countsketches[key] = sketcher
        return sketcher.to(device)

    def _muon_right_preconditioner(
        self,
        momentum_matrix: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        M = momentum_matrix.to(torch.float32)
        MtM = M.T @ M
        d_in = MtM.size(0)
        MtM = MtM + eps * torch.eye(d_in, device=M.device, dtype=torch.float32)
        eigvals, eigvecs = torch.linalg.eigh(MtM)
        inv_sqrt = eigvals.clamp(min=eps).rsqrt()
        return (eigvecs * inv_sqrt.unsqueeze(0)) @ eigvecs.T

    def _countsketch_outer(
        self,
        layer_name: str,
        activations: torch.Tensor,
        backprops: torch.Tensor,
        preconditioner: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        activations = activations.detach().to(torch.float32)
        backprops = backprops.detach().to(torch.float32)
        if activations.dim() < 2 or backprops.dim() < 2:
            raise ValueError("Ghost factors must have shape (B, ..., d)")
        if activations.size(0) != backprops.size(0):
            raise ValueError(
                "Activation/backprop batch dimensions differ: "
                f"{tuple(activations.shape)} vs {tuple(backprops.shape)}"
            )

        if activations.dim() == 2:
            activations = activations.unsqueeze(1)
        else:
            activations = activations.reshape(
                activations.size(0), -1, activations.size(-1)
            )
        if backprops.dim() == 2:
            backprops = backprops.unsqueeze(1)
        else:
            backprops = backprops.reshape(
                backprops.size(0), -1, backprops.size(-1)
            )
        if activations.size(1) != backprops.size(1):
            raise ValueError(
                "Activation/backprop token dimensions differ: "
                f"{tuple(activations.shape)} vs {tuple(backprops.shape)}"
            )

        B, n_tokens, d_in = activations.shape
        d_out = backprops.size(-1)
        sketcher = self._get_countsketch(
            layer_name,
            d_in=d_in,
            d_out=d_out,
            device=activations.device,
        )
        token_block = max(1, int(getattr(self.config, "countsketch_token_block", 128)))
        out = torch.zeros(
            B,
            int(self.config.sketch_dim),
            device=activations.device,
            dtype=torch.float32,
        )
        for start in range(0, n_tokens, token_block):
            end = min(start + token_block, n_tokens)
            block_tokens = end - start
            flat_sketches = sketcher.sketch_outer(
                activations[:, start:end, :].reshape(B * block_tokens, d_in),
                backprops[:, start:end, :].reshape(B * block_tokens, d_out),
                preconditioner,
            )
            out += flat_sketches.reshape(B, block_tokens, -1).sum(dim=1)
        return out

    def _precondition_activations(
        self,
        activations: torch.Tensor,
        preconditioner: torch.Tensor,
    ) -> torch.Tensor:
        activations = activations.detach().to(torch.float32)
        P = preconditioner.to(device=activations.device, dtype=torch.float32)
        original_shape = activations.shape
        flat = activations.reshape(-1, original_shape[-1]) @ P
        return flat.reshape(*original_shape[:-1], P.size(-1))

    def _candidate_countsketch_layer(
        self,
        layer_name: str,
        activations: torch.Tensor,
        backprops: torch.Tensor,
    ) -> torch.Tensor:
        layer = self._target_layers[layer_name]
        precond_info = extract_optimizer_preconditioner(
            self.optimizer, layer.weight, self.opt_name
        )
        kind = precond_info["type"]

        if kind == "adam_family":
            precond = precond_info["preconditioner"].reshape(layer.weight.shape)
            return self._countsketch_outer(layer_name, activations, backprops, precond)
        if kind == "lion":
            precond = precond_info["sign_ut"].reshape(layer.weight.shape)
            return self._countsketch_outer(layer_name, activations, backprops, precond)
        if kind == "sophia":
            precond = precond_info["hessian_diag"].reshape(layer.weight.shape)
            return self._countsketch_outer(layer_name, activations, backprops, precond)
        if kind == "cadamw":
            precond = (
                precond_info["preconditioner"].reshape(layer.weight.shape)
                * precond_info["mask"].reshape(layer.weight.shape)
                * precond_info["scale"]
            )
            return self._countsketch_outer(layer_name, activations, backprops, precond)
        if kind == "muon":
            P = self._muon_right_preconditioner(precond_info["momentum_matrix"])
            a_pre = self._precondition_activations(activations, P)
            return self._countsketch_outer(layer_name, a_pre, backprops, None)

        # SGD and any optimizer state that has not been initialized yet.
        return self._countsketch_outer(layer_name, activations, backprops, None)

    def build_candidate_sketches(
        self,
        candidate_activations: Dict[str, torch.Tensor],
        candidate_backprops: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Build OPUS phi^(t,r)(z) CountSketch features for candidates."""
        if not self._proxy_sketches:
            raise RuntimeError("Validation proxy sketches have not been computed")

        sketches: Dict[str, torch.Tensor] = {}
        for layer_name in candidate_activations:
            if layer_name not in self._proxy_sketches:
                continue
            sketches[layer_name] = self._candidate_countsketch_layer(
                layer_name,
                candidate_activations[layer_name],
                candidate_backprops[layer_name],
            )

        if not sketches:
            raise RuntimeError("No layers found for CountSketch scoring")
        return sketches

    def score_candidate_sketches(
        self,
        candidate_sketches: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute sum_r <phi_z^(t,r), psi_proxy^(t,r)> in sketch space."""
        total_scores: Optional[torch.Tensor] = None
        for layer_name, phi in candidate_sketches.items():
            proxy = self._proxy_sketches.get(layer_name)
            if proxy is None:
                continue
            layer_scores = (phi * proxy.unsqueeze(0)).sum(dim=-1)
            layer_scores = torch.nan_to_num(
                layer_scores, nan=0.0, posinf=1e9, neginf=-1e9
            )
            total_scores = layer_scores if total_scores is None else total_scores + layer_scores

        if total_scores is None:
            raise RuntimeError("No layers found for CountSketch scoring")
        return torch.nan_to_num(total_scores, nan=0.0, posinf=1e9, neginf=-1e9)

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

        if self.config.use_countsketch:
            accum_sketch: Dict[str, torch.Tensor] = {}
            total_examples = 0

            for val_x, val_y in val_batches:
                self.start_capture()
                model.zero_grad(set_to_none=True)
                with type_ctx:
                    outputs = model(val_x, targets=val_y)
                outputs["loss"].backward()
                self.stop_capture()

                for name in self._layer_activations:
                    if name not in self._layer_backprops:
                        continue
                    sketches = self._countsketch_outer(
                        name,
                        self._layer_activations[name],
                        self._layer_backprops[name],
                        preconditioner=None,
                    )
                    layer_sum = sketches.sum(dim=0)
                    if name not in accum_sketch:
                        accum_sketch[name] = layer_sum
                    else:
                        accum_sketch[name] += layer_sum

                total_examples += val_x.size(0)
                self._layer_activations.clear()
                self._layer_backprops.clear()
                del outputs

            if total_examples <= 0 or not accum_sketch:
                raise RuntimeError("No Ghost factors were captured for proxy sketches")

            self._proxy_sketches = {
                name: accum_sketch[name] / float(total_examples)
                for name in accum_sketch
            }
            self._val_layer_activations = {}
            self._val_layer_backprops = {}
            self._last_candidate_sketches = None
            model.zero_grad(set_to_none=True)
            model.train()
            return

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
        if self.config.use_countsketch:
            if candidate_activations:
                self._last_candidate_sketches = self.build_candidate_sketches(
                    candidate_activations,
                    candidate_backprops,
                )
            elif self._last_candidate_sketches is None:
                raise RuntimeError("No candidate sketches were provided for scoring")
            return self.score_candidate_sketches(self._last_candidate_sketches)

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

    def _normalize_selection_utility(
        self,
        utility: torch.Tensor,
        available_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        utility = torch.nan_to_num(
            utility, nan=0.0, posinf=1e9, neginf=-1e9
        )
        active = utility.masked_select(available_mask)
        if active.numel() <= 0:
            return utility, 0.0
        mean = active.mean()
        std = active.std(unbiased=False) if active.numel() > 1 else active.new_zeros(())
        normalized = (utility - mean) / std.clamp_min(1e-8)
        normalized = torch.nan_to_num(
            normalized, nan=0.0, posinf=1e9, neginf=-1e9
        )
        return normalized, float(std.item())

    def _masked_softmax_entropy(
        self,
        scores: torch.Tensor,
        available_mask: torch.Tensor,
        temperature: float,
        reference_count: int,
    ) -> Tuple[float, float]:
        n_avail = int(available_mask.sum().item())
        if n_avail <= 1:
            return 0.0, 0.0
        available_scores = scores.masked_select(available_mask)
        max_score = available_scores.max()
        if not torch.isfinite(max_score):
            entropy = math.log(float(n_avail))
        else:
            scaled = ((available_scores - max_score) / max(temperature, 1e-6)).clamp(
                min=-50.0, max=50.0
            )
            probs = torch.exp(scaled)
            probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
            total = probs.sum()
            if not torch.isfinite(total) or total.item() < 1e-20:
                entropy = math.log(float(n_avail))
            else:
                probs = probs / total
                entropy = float((-(probs * torch.log(probs + 1e-10)).sum()).item())
                if not math.isfinite(entropy):
                    entropy = 0.0
        denom = math.log(float(max(reference_count, 2)))
        return entropy, entropy / denom if denom > 0 else 0.0

    def _selected_score_percentile(
        self,
        alignment_scores: torch.Tensor,
        selected_idx: torch.Tensor,
    ) -> float:
        if alignment_scores.numel() <= 1 or selected_idx.numel() == 0:
            return 0.0
        ranks = torch.empty_like(alignment_scores, dtype=torch.float32)
        order = torch.argsort(alignment_scores)
        ranks[order] = torch.arange(
            alignment_scores.numel(),
            device=alignment_scores.device,
            dtype=torch.float32,
        )
        percentiles = ranks[selected_idx] / float(alignment_scores.numel() - 1)
        return float(percentiles.mean().item())

    def _record_selection_stats(
        self,
        alignment_scores: torch.Tensor,
        selected_idx: torch.Tensor,
        eta: float,
        temperature: float,
        utility_stds: List[float],
        entropies: List[float],
        entropy_over_log_candidates: List[float],
    ):
        with torch.no_grad():
            clean_scores = torch.nan_to_num(
                alignment_scores, nan=0.0, posinf=1e9, neginf=-1e9
            )
            score_std = (
                float(clean_scores.std(unbiased=False).item())
                if clean_scores.numel() > 1 else 0.0
            )
            selected_scores = clean_scores[selected_idx] if selected_idx.numel() else clean_scores[:0]
            entropy = (
                sum(entropies) / len(entropies)
                if entropies else 0.0
            )
            entropy_over_log = (
                sum(entropy_over_log_candidates) / len(entropy_over_log_candidates)
                if entropy_over_log_candidates else 0.0
            )
            utility_std = (
                sum(utility_stds) / len(utility_stds)
                if utility_stds else 0.0
            )

            self.selection_stats["entropy"].append(entropy)
            self.selection_stats["mean_score"].append(float(clean_scores.mean().item()))
            self.selection_stats["std_score"].append(score_std)
            self.selection_stats["utility_std"].append(utility_std)
            self.selection_stats["eta_score_std_over_temperature"].append(
                float(abs(eta) * score_std / max(temperature, 1e-6))
            )
            self.selection_stats["effective_entropy_over_log_candidates"].append(
                entropy_over_log
            )
            self.selection_stats["selected_score_mean"].append(
                float(selected_scores.mean().item()) if selected_scores.numel() else 0.0
            )
            self.selection_stats["candidate_score_mean"].append(
                float(clean_scores.mean().item())
            )
            self.selection_stats["selected_score_percentile"].append(
                self._selected_score_percentile(clean_scores, selected_idx)
            )

    def _select_batch_from_sketches(
        self,
        candidate_sketches: Dict[str, torch.Tensor],
        alignment_scores: torch.Tensor,
        n_select: int,
        eta: float,
        lambda_r: float,
    ) -> torch.Tensor:
        """
        OPUS Eq. 25 selection in sketch space.

        U_z ≈ η Σ_r <phi_z^r, psi_proxy^r>
              - η² λ_r Σ_r <phi_z^r, Phi^r>,
        where Phi^r is the running sum of already selected sketches.
        """
        B_cand = alignment_scores.size(0)
        device = alignment_scores.device
        temperature = self.config.temperature
        alignment_scores = torch.nan_to_num(
            alignment_scores, nan=0.0, posinf=1e9, neginf=-1e9
        )

        selected: List[int] = []
        available_mask = torch.ones(B_cand, dtype=torch.bool, device=device)
        history = {
            name: torch.zeros_like(sketches[0])
            for name, sketches in candidate_sketches.items()
        }
        utility_stds: List[float] = []
        entropies: List[float] = []
        entropy_over_log_candidates: List[float] = []

        for round_idx in range(n_select):
            redundancy = torch.zeros(B_cand, device=device, dtype=torch.float32)
            if round_idx > 0:
                for name, sketches in candidate_sketches.items():
                    redundancy = redundancy + (
                        sketches * history[name].unsqueeze(0)
                    ).sum(dim=-1)

            redundancy = torch.nan_to_num(
                redundancy, nan=0.0, posinf=1e9, neginf=-1e9
            )
            utility = eta * alignment_scores - (eta ** 2) * lambda_r * redundancy
            utility, utility_std = self._normalize_selection_utility(
                utility, available_mask
            )
            entropy, entropy_over_log = self._masked_softmax_entropy(
                utility,
                available_mask,
                temperature,
                reference_count=B_cand,
            )
            utility_stds.append(utility_std)
            entropies.append(entropy)
            entropy_over_log_candidates.append(entropy_over_log)
            idx = self._safe_boltzmann_sample(utility, available_mask, temperature)

            selected.append(idx)
            available_mask[idx] = False
            for name, sketches in candidate_sketches.items():
                history[name] = history[name] + sketches[idx]

        selected_tensor = torch.tensor(selected, device=device, dtype=torch.long)
        self._record_selection_stats(
            alignment_scores,
            selected_tensor,
            eta=eta,
            temperature=temperature,
            utility_stds=utility_stds,
            entropies=entropies,
            entropy_over_log_candidates=entropy_over_log_candidates,
        )
        return selected_tensor

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
        if self.config.use_countsketch:
            if self._last_candidate_sketches is None:
                self._last_candidate_sketches = self.build_candidate_sketches(
                    candidate_activations,
                    candidate_backprops,
                )
            return self._select_batch_from_sketches(
                self._last_candidate_sketches,
                alignment_scores=alignment_scores,
                n_select=n_select,
                eta=eta,
                lambda_r=lambda_r,
            )

        B_cand = alignment_scores.size(0)
        device = alignment_scores.device
        temperature = self.config.temperature

        alignment_scores = torch.nan_to_num(
            alignment_scores, nan=0.0, posinf=1e9, neginf=-1e9
        )

        selected: List[int] = []
        available_mask = torch.ones(B_cand, dtype=torch.bool, device=device)
        utility_stds: List[float] = []
        entropies: List[float] = []
        entropy_over_log_candidates: List[float] = []

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

            utility = eta * alignment_scores - (eta ** 2) * lambda_r * redundancy
            utility, utility_std = self._normalize_selection_utility(
                utility, available_mask
            )
            entropy, entropy_over_log = self._masked_softmax_entropy(
                utility,
                available_mask,
                temperature,
                reference_count=B_cand,
            )
            utility_stds.append(utility_std)
            entropies.append(entropy)
            entropy_over_log_candidates.append(entropy_over_log)

            idx = self._safe_boltzmann_sample(utility, available_mask, temperature)

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

        selected_tensor = torch.tensor(selected, device=device, dtype=torch.long)
        self._record_selection_stats(
            alignment_scores,
            selected_tensor,
            eta=eta,
            temperature=temperature,
            utility_stds=utility_stds,
            entropies=entropies,
            entropy_over_log_candidates=entropy_over_log_candidates,
        )
        return selected_tensor

    def get_selection_summary(self) -> Dict:
        if not self.selection_stats["entropy"]:
            return {}
        summary = {}
        for name, values in self.selection_stats.items():
            if values:
                summary[f"mean_{name}"] = sum(values) / len(values)
        return summary
