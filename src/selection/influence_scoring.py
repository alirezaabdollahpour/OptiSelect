"""
Optimizer-aware influence scoring for online in-batch data selection.

Implements the frozen-state transformation operators O_t from Section 4.3
of the OptiSelect paper. Each optimizer family has its own scoring function
that computes ⟨O_t(g_z), g_V⟩_F efficiently via the Ghost trick.

Paper references:
  Eq. (4):  U(z,t) ≈ η ⟨O_t(g_z), g_V⟩ - η² ⟨O_t(g_z), G_t⟩   (alignment - redundancy)
  Eq. (7):  AdamW operator
  Eq. (8):  AdEMAMix operator (same as AdamW)
  Eq. (11): Sophia operator with clip
  Eq. (13): Lion operator
  Eq. (15): Muon right-preconditioner
  Eq. (18): SOAP rotated operator
  Eq. (22): GaLore projected operator
  Eq. (24): C-AdamW masked operator
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class InfluenceConfig:
    """Configuration for influence scoring."""
    sketch_dim: int = 1024
    temperature: float = 0.1
    redundancy_weight: float = 1.0  # λ_r from Paper Eq. 4
    use_countsketch: bool = False   # Disabled by default (exact at 124M scale)


class CountSketch:
    """
    CountSketch for optional compression of Ghost factors (Paper Section 4.4).

    For 124M-scale experiments the exact inner product is cheap enough that
    CountSketch is unnecessary. We keep this class as a utility for future
    larger-scale runs but do not apply it by default.
    """
    def __init__(self, d: int, m: int, seed: int = 42):
        gen = torch.Generator()
        gen.manual_seed(seed)
        self.m = m
        self.hash_indices = torch.randint(0, m, (d,), generator=gen)
        self.signs = torch.randint(0, 2, (d,), generator=gen).float() * 2 - 1

    def to(self, device):
        self.hash_indices = self.hash_indices.to(device)
        self.signs = self.signs.to(device)
        return self

    def sketch(self, x: torch.Tensor) -> torch.Tensor:
        signed = x * self.signs
        result = torch.zeros(*x.shape[:-1], self.m, device=x.device, dtype=x.dtype)
        result.scatter_add_(-1, self.hash_indices.expand_as(signed), signed)
        return result


# =============================================================================
# Ghost inner product helper
# =============================================================================

def _ghost_inner_product(
    cand_a: torch.Tensor,       # (B, seq, d_in)
    cand_b: torch.Tensor,       # (B, seq, d_out)
    D_V: torch.Tensor,          # (d_out, d_in) preconditioned val direction
) -> torch.Tensor:
    """
    Compute per-sample Ghost-factored Frobenius inner product:
        <sum_t b_z[t] a_z[t]^T, D_V>_F = sum_t b_z[t]^T (D_V @ a_z[t])

    Returns (B,) tensor of inner products.
    """
    D_V_a = torch.einsum('bsi,oi->bso', cand_a, D_V)
    return (cand_b * D_V_a).sum(dim=(1, 2))


# =============================================================================
# Diagonal adaptive operators (AdamW, AdEMAMix, ADOPT, Adam-mini, Sophia)
# =============================================================================

def compute_adam_family_scores(
    activations: torch.Tensor,
    backprops: torch.Tensor,
    val_activations: torch.Tensor,
    val_backprops: torch.Tensor,
    preconditioner: torch.Tensor,   # (d_out, d_in): c_t = 1/(√v̂ + ε)
) -> torch.Tensor:
    """
    Paper Eq. (7), (28): Adam-family influence score.

    O_t^Adam(g) = c_t ⊙ g, where c_t = 1/(√v̂_t + ε).

    Using Ghost factorization g_V ≈ b_V a_V^T and G_z = b_z a_z^T:
        <O_t^Adam(G_z), G_V>_F = <c_t ⊙ G_z, G_V>_F
                               = sum_t b_z[t]^T (D_V @ a_z[t])
    where D_V = c_t ⊙ (b_V a_V^T).
    """
    C = preconditioner.to(torch.float32)

    val_a = val_activations.reshape(-1, val_activations.size(-1)).to(torch.float32).mean(0)
    val_b = val_backprops.reshape(-1, val_backprops.size(-1)).to(torch.float32).mean(0)

    D_V = C * torch.outer(val_b, val_a)

    return _ghost_inner_product(
        activations.to(torch.float32),
        backprops.to(torch.float32),
        D_V,
    )


def compute_sophia_scores(
    activations: torch.Tensor,
    backprops: torch.Tensor,
    val_activations: torch.Tensor,
    val_backprops: torch.Tensor,
    hessian_precond: torch.Tensor,  # (d_out, d_in): 1/max(ρh_t, ε)
) -> torch.Tensor:
    """
    Paper Eq. (11), (29): Sophia influence score (linearized).

    O_t^Sophia(x) = clip(x / max(ρ h_t, ε), 1).

    We use the LINEARIZED operator x / max(ρ h_t, ε) for scoring.
    The clip-to-1 is verified empirically to be inactive on >95% of
    coordinates during stable training (Paper Appendix B).
    Structurally identical to Adam scoring with h in place of √v̂.
    """
    return compute_adam_family_scores(
        activations, backprops, val_activations, val_backprops, hessian_precond
    )


# =============================================================================
# Sign-based operator (Lion, Signum, SignSGD)
# =============================================================================

def compute_lion_scores(
    activations: torch.Tensor,
    backprops: torch.Tensor,
    val_activations: torch.Tensor,
    val_backprops: torch.Tensor,
    sign_ut: torch.Tensor,          # (d_out, d_in): s_t = sign(m_{t-1})
) -> torch.Tensor:
    """
    Paper Eq. (14), Theorem 1(i): Lion influence score (linearized).

    O_t^Lion(g) = sign((1-β₁)g + β₁ m_{t-1}).

    For frozen-state scoring, we use the leading-order approximation
    treating s_t = sign(m_{t-1}) as a fixed diagonal preconditioner:
        U_z^Lion,lin = η ⟨g_z, s_t ⊙ g_V⟩

    This is the quantity analyzed in Theorem 1; it provides an upper bound
    on the true Lion score variance (since sign(·) clips magnitudes further).
    """
    S = sign_ut.to(torch.float32)

    val_a = val_activations.reshape(-1, val_activations.size(-1)).to(torch.float32).mean(0)
    val_b = val_backprops.reshape(-1, val_backprops.size(-1)).to(torch.float32).mean(0)

    D_V = S * torch.outer(val_b, val_a)

    return _ghost_inner_product(
        activations.to(torch.float32),
        backprops.to(torch.float32),
        D_V,
    )


# =============================================================================
# Matrix-based operators (Muon, SOAP)
# =============================================================================

def compute_muon_scores(
    activations: torch.Tensor,
    backprops: torch.Tensor,
    val_activations: torch.Tensor,
    val_backprops: torch.Tensor,
    momentum_matrix: torch.Tensor,  # M_t: (d_out, d_in)
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Paper Eq. (15), (31): Muon right-preconditioned influence score.

    O_t^Muon(G_z) ≈ G_z (M_t^T M_t + εI)^{-1/2}

    This equalizes the singular values of M_t, replacing σ_i → 1/σ_i on the
    row space of M_t. The Ghost-factored inner product becomes:
        <G_z P, b_V a_V^T>_F = (a_z^T P a_V)(b_z^T b_V)
    where P = (M_t^T M_t + εI)^{-1/2} is computed once per step.
    """
    M = momentum_matrix.to(torch.float32)

    MtM = M.T @ M
    d_in = MtM.size(0)
    MtM = MtM + eps * torch.eye(d_in, device=M.device, dtype=torch.float32)

    eigvals, eigvecs = torch.linalg.eigh(MtM)
    inv_sqrt = eigvals.clamp(min=eps).rsqrt()
    P = (eigvecs * inv_sqrt.unsqueeze(0)) @ eigvecs.T

    val_a = val_activations.reshape(-1, val_activations.size(-1)).to(torch.float32).mean(0)
    val_b = val_backprops.reshape(-1, val_backprops.size(-1)).to(torch.float32).mean(0)

    P_aV = P @ val_a

    cand_a = activations.to(torch.float32)
    cand_b = backprops.to(torch.float32)

    a_dot = torch.einsum('bsi,i->bs', cand_a, P_aV).sum(dim=1)
    b_dot = torch.einsum('bso,o->bs', cand_b, val_b).sum(dim=1)

    return a_dot * b_dot


def compute_soap_scores(
    activations: torch.Tensor,
    backprops: torch.Tensor,
    val_activations: torch.Tensor,
    val_backprops: torch.Tensor,
    U_L: torch.Tensor,              # (d_out, d_out) left eigenbasis
    U_R: torch.Tensor,              # (d_in, d_in) right eigenbasis
    v_rotated: torch.Tensor,        # (d_out, d_in) variance in rotated basis
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Paper Eq. (18), (32): SOAP rotated diagonal influence score.

    O_t^SOAP(G_z) = U_L [(U_L^T G_z U_R) ⊘ (√ṽ + ε)] U_R^T

    Rotated Ghost factors: ã(z) = U_R^T a(z), b̃(z) = U_L^T b(z).
    The inner product reduces to the Adam-family form in the rotated basis:
        <O_t^SOAP(G_z), G_V>_F = ã(z)^T D̃_V^T b̃(z)
    with D̃_V = (1/(√ṽ+ε)) ⊙ b̃_V ã_V^T.
    """
    U_L = U_L.to(torch.float32)
    U_R = U_R.to(torch.float32)
    v_rot = v_rotated.to(torch.float32)

    C_rot = 1.0 / (v_rot.sqrt() + eps)

    val_a = val_activations.reshape(-1, val_activations.size(-1)).to(torch.float32).mean(0)
    val_b = val_backprops.reshape(-1, val_backprops.size(-1)).to(torch.float32).mean(0)
    a_V_rot = U_R.T @ val_a
    b_V_rot = U_L.T @ val_b

    D_V_rot = C_rot * torch.outer(b_V_rot, a_V_rot)

    cand_a = activations.to(torch.float32)
    cand_b = backprops.to(torch.float32)

    cand_a_rot = torch.einsum('bsi,ji->bsj', cand_a, U_R)
    cand_b_rot = torch.einsum('bso,jo->bsj', cand_b, U_L)

    return _ghost_inner_product(cand_a_rot, cand_b_rot, D_V_rot)


# =============================================================================
# Memory-efficient operators (GaLore, C-AdamW)
# =============================================================================

def compute_galore_scores(
    activations: torch.Tensor,
    backprops: torch.Tensor,
    val_activations: torch.Tensor,
    val_backprops: torch.Tensor,
    Pi_L: torch.Tensor,             # (d_out, r) left projector
    Pi_R: torch.Tensor,             # (d_in, r) right projector
    v_projected: torch.Tensor,      # (r, r) variance in projected space
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Paper Eq. (22), (33): GaLore projected influence score.

    O_t^GaLore(G_z) = Π_L [(Π_L^T G_z Π_R) ⊘ (√Ṽ + ε)] Π_R^T

    Projects Ghost factors into rank-r subspace. For r=256 this is much
    cheaper than full-rank Adam.
    """
    Pi_L = Pi_L.to(torch.float32)
    Pi_R = Pi_R.to(torch.float32)
    v_proj = v_projected.to(torch.float32)

    C_proj = 1.0 / (v_proj.sqrt() + eps)

    val_a = val_activations.reshape(-1, val_activations.size(-1)).to(torch.float32).mean(0)
    val_b = val_backprops.reshape(-1, val_backprops.size(-1)).to(torch.float32).mean(0)
    a_V_proj = Pi_R.T @ val_a
    b_V_proj = Pi_L.T @ val_b

    D_V_proj = C_proj * torch.outer(b_V_proj, a_V_proj)

    cand_a = activations.to(torch.float32)
    cand_b = backprops.to(torch.float32)
    cand_a_proj = torch.einsum('bsi,ir->bsr', cand_a, Pi_R)
    cand_b_proj = torch.einsum('bso,or->bsr', cand_b, Pi_L)

    return _ghost_inner_product(cand_a_proj, cand_b_proj, D_V_proj)


def compute_cadamw_scores(
    activations: torch.Tensor,
    backprops: torch.Tensor,
    val_activations: torch.Tensor,
    val_backprops: torch.Tensor,
    preconditioner: torch.Tensor,   # c_t: (d_out, d_in)
    mask: torch.Tensor,             # φ_t: (d_out, d_in), binary
    scale: float,                   # d / (||φ||_0 + 1)
) -> torch.Tensor:
    """
    Paper Eq. (24): C-AdamW masked influence score.

    O_t^CAdamW(g) ≈ (d / (||φ||_0 + 1)) · φ_t ⊙ c_t ⊙ g

    Keeps AdamW's diagonal geometry on the active coordinates, suppresses
    directions where the preconditioned sign conflicts with the gradient.
    """
    C = (preconditioner.to(torch.float32) *
         mask.to(torch.float32) * scale)

    val_a = val_activations.reshape(-1, val_activations.size(-1)).to(torch.float32).mean(0)
    val_b = val_backprops.reshape(-1, val_backprops.size(-1)).to(torch.float32).mean(0)

    D_V = C * torch.outer(val_b, val_a)

    return _ghost_inner_product(
        activations.to(torch.float32),
        backprops.to(torch.float32),
        D_V,
    )


# =============================================================================
# Preconditioner extraction from optimizer state
# =============================================================================

def extract_optimizer_preconditioner(
    optimizer: torch.optim.Optimizer,
    param: nn.Parameter,
    opt_name: str,
) -> Dict:
    """
    Extract the frozen-state operator parameters for `param`.

    Returns a dict keyed by optimizer type. Each scoring branch in
    OptiSelectEngine.score_candidates handles its own type.
    """
    state = optimizer.state.get(param, {})

    # ---- Diagonal adaptive family ----
    if opt_name in ("adamw", "ademamix", "adam-mini"):
        if "exp_avg_sq" in state:
            v = state["exp_avg_sq"]
            step = state.get("step", 1)
            if isinstance(step, torch.Tensor):
                step = int(step.item())
            beta2 = _get_beta2(optimizer, param)
            bc2 = 1.0 - beta2 ** step if step > 0 else 1.0
            v_hat = v / bc2
            eps = _get_eps(optimizer)
            precond = 1.0 / (v_hat.sqrt() + eps)
            return {"type": "adam_family", "preconditioner": precond}
        return {"type": "sgd"}

    # ---- ADOPT (delayed variance, Paper Section 4.3.1) ----
    elif opt_name == "adopt":
        if "exp_avg_sq" in state:
            v_prev = state.get("_prev_exp_avg_sq", state["exp_avg_sq"])
            step = state.get("step", 1)
            if isinstance(step, torch.Tensor):
                step = int(step.item())
            beta2 = _get_beta2(optimizer, param)
            bc2 = 1.0 - beta2 ** max(step - 1, 1)
            v_hat_prev = v_prev / bc2
            eps = _get_eps(optimizer)
            precond = 1.0 / torch.clamp(v_hat_prev.sqrt(), min=eps)
            return {"type": "adam_family", "preconditioner": precond}
        return {"type": "sgd"}

    # ---- MARS (AdamW-style scoring on raw gradients, Paper Remark 4) ----
    elif opt_name == "mars":
        scoring_v = getattr(param, '_scoring_v', None)
        if scoring_v is not None:
            eps = 1e-8
            precond = 1.0 / (scoring_v.sqrt() + eps)
            return {"type": "adam_family", "preconditioner": precond}
        return {"type": "sgd"}

    # ---- Sign-based (Lion, Signum, SignSGD) ----
    elif opt_name in ("lion", "signum", "signsgd"):
        if "exp_avg" in state:
            # Paper Theorem 1(i): s_t = sign(m_{t-1})
            # After opt.step(), state["exp_avg"] = m_t. On the NEXT scoring pass,
            # m_t plays the role of m_{t-1}, so current state is correct.
            m = state["exp_avg"]
            return {"type": "lion", "sign_ut": torch.sign(m)}
        return {"type": "sgd"}

    # ---- Sophia (curvature-aware, Paper Eq. 11) ----
    elif opt_name == "sophiag":
        if "hessian" in state:
            h = state["hessian"]
            rho = 0.04  # Paper Appendix C
            for group in optimizer.param_groups:
                if any(p is param for p in group["params"]):
                    rho = group.get("rho", 0.04)
                    break
            eps = 1e-12
            denom = torch.clamp(h * rho, min=eps)
            precond = 1.0 / denom
            return {"type": "sophia", "hessian_diag": precond}
        return {"type": "sgd"}

    # ---- Muon (matrix, Paper Eq. 15) ----
    elif opt_name in ("muon", "d-muon"):
        if "momentum_buffer" in state:
            M = state["momentum_buffer"]
            if M.dim() == 2:
                return {"type": "muon", "momentum_matrix": M}
        return {"type": "sgd"}

    # ---- SOAP (rotated Adam, Paper Eq. 18) ----
    elif opt_name == "soap":
        # SOAP stores Q = [Q_L, Q_R] (Kronecker eigenbasis) and exp_avg_sq
        # in the rotated basis.
        if "Q" in state and "exp_avg_sq" in state:
            Q = state["Q"]
            if isinstance(Q, list) and len(Q) >= 2:
                U_L = Q[0]
                U_R = Q[1]
                v_rot = state["exp_avg_sq"]
                step = state.get("step", 1)
                if isinstance(step, torch.Tensor):
                    step = int(step.item())
                beta2 = _get_beta2(optimizer, param)
                bc2 = 1.0 - beta2 ** step if step > 0 else 1.0
                return {
                    "type": "soap",
                    "U_L": U_L,
                    "U_R": U_R,
                    "v_rotated": v_rot / bc2,
                }
        return {"type": "sgd"}

    # ---- GaLore (low-rank projection, Paper Eq. 22) ----
    elif opt_name == "galore":
        if "projector" in state and "exp_avg_sq" in state:
            proj = state["projector"]
            Pi_L = getattr(proj, 'ortho_matrix_L', None)
            Pi_R = getattr(proj, 'ortho_matrix_R', None)
            if Pi_L is not None and Pi_R is not None:
                v_proj = state["exp_avg_sq"]
                step = state.get("step", 1)
                if isinstance(step, torch.Tensor):
                    step = int(step.item())
                beta2 = _get_beta2(optimizer, param)
                bc2 = 1.0 - beta2 ** step if step > 0 else 1.0
                return {
                    "type": "galore",
                    "Pi_L": Pi_L,
                    "Pi_R": Pi_R,
                    "v_projected": v_proj / bc2,
                }
        return {"type": "sgd"}

    # ---- C-AdamW (cautious mask, Paper Eq. 24) ----
    elif opt_name == "c-adamw":
        if "exp_avg_sq" in state and param.grad is not None:
            v = state["exp_avg_sq"]
            m = state.get("exp_avg", torch.zeros_like(param))
            step = state.get("step", 1)
            if isinstance(step, torch.Tensor):
                step = int(step.item())
            beta1 = _get_beta1(optimizer, param)
            beta2 = _get_beta2(optimizer, param)
            bc1 = 1.0 - beta1 ** step if step > 0 else 1.0
            bc2 = 1.0 - beta2 ** step if step > 0 else 1.0
            m_hat = m / bc1
            v_hat = v / bc2
            eps = _get_eps(optimizer)
            precond = 1.0 / (v_hat.sqrt() + eps)

            u = m_hat * precond
            mask = (u * param.grad > 0).float()
            d_total = float(mask.numel())
            active = mask.sum().item()
            scale = d_total / (active + 1)
            return {
                "type": "cadamw",
                "preconditioner": precond,
                "mask": mask,
                "scale": scale,
            }
        return {"type": "sgd"}

    return {"type": "sgd"}


def _get_beta1(optimizer, param):
    for group in optimizer.param_groups:
        if any(p is param for p in group["params"]):
            betas = group.get("betas", (0.9, 0.999))
            return betas[0] if len(betas) > 0 else 0.9
    return 0.9


def _get_beta2(optimizer, param):
    for group in optimizer.param_groups:
        if any(p is param for p in group["params"]):
            betas = group.get("betas", (0.9, 0.999))
            return betas[1] if len(betas) > 1 else 0.999
    return 0.999


def _get_eps(optimizer):
    for group in optimizer.param_groups:
        return group.get("eps", 1e-8)
    return 1e-8


# =============================================================================
# Boltzmann selection
# =============================================================================

def boltzmann_selection(
    scores: torch.Tensor,
    n_select: int,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Boltzmann sampling without replacement (Paper Eq. 34).

    p_t(z) = exp(U(z,t) / τ) / Σ exp(U(z',t) / τ)
    """
    scores = scores - scores.max()
    probs = torch.softmax(scores / temperature, dim=0)
    return torch.multinomial(probs, n_select, replacement=False)