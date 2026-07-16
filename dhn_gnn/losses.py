"""Physics loss (spec §4).

L_phys = sum_k  gamma^(K-1-k) * mean( (B_internal . phi(B^T m_k))^2 )

i.e. the discounted sum of squared internal-loop residuals over the K unrolled
steps, in Pa^2. Mirrors the paper's discounted mismatch loss
(GNSMsg_SelfAttention_armijo.py:428-430) with the DHN governing operator.
"""

import torch


def discounted_physics_loss(residual_terms, gamma: float = 0.9):
    """
    residual_terms: list of per-step residual tensors r_k = B_internal . phi(mdot_k),
                    each shape (..., L_int). Length K, ordered k = 0 .. K-1.
    Returns a scalar (Pa^2), later-step residuals weighted more heavily by gamma.
    """
    K = len(residual_terms)
    loss = residual_terms[0].new_zeros(())
    for k, r in enumerate(residual_terms):
        loss = loss + (gamma ** (K - 1 - k)) * (r ** 2).mean()
    return loss
