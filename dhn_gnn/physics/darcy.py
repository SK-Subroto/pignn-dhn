"""Differentiable PyTorch port of PyDHN's pipe hydraulics (the *known operator*).

This is a faithful reimplementation of PyDHN's NumPy functions so that gradients
can flow through the pressure-flow map in the unrolled solver. It is *validated*
against PyDHN in float64 (see tests/run_gates.py, Gate 1).

Ports:
  - reynolds            <- pydhn.fluids.dimensionless_numbers.compute_reynolds
  - friction_factor     <- pydhn.components.base_components_hydraulics.compute_friction_factor
  - phi / dphi_dmdot    <- pydhn.components.base_components_hydraulics.compute_dp_pipe

All functions are elementwise over tensors of matching shape and preserve dtype
(use float64 for validation). Reynolds/friction use PyDHN's `safe_divide`
convention: division by zero yields 0 (not NaN/inf).
"""

import math

import torch

_PI = math.pi


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """Elementwise num/den, returning 0 where den == 0 (mirrors pydhn.safe_divide)."""
    den_is_zero = den == 0
    safe_den = torch.where(den_is_zero, torch.ones_like(den), den)
    out = num / safe_den
    return torch.where(den_is_zero, torch.zeros_like(out), out)


def reynolds(mdot: torch.Tensor, diameter: torch.Tensor, mu: float) -> torch.Tensor:
    """Re = 4|mdot| / (pi * d * mu). Port of compute_reynolds."""
    num = 4.0 * mdot.abs()
    den = _PI * diameter * mu
    return _safe_div(num, den)


def _turbulent_fd(Re, diameter, roughness):
    """Darcy friction factor, turbulent branch (Haaland-form, pydhn :29-42)."""
    rel_roughness = _safe_div(roughness, diameter * 1000.0)
    div = _safe_div(torch.as_tensor(6.9, dtype=Re.dtype, device=Re.device), Re)
    log = torch.log10((rel_roughness / 3.7) ** 1.11 + div)
    return (1.8 * log) ** -2


def _transition_fd(Re, diameter, roughness):
    """
    Darcy friction factor, transitional branch (non-affine), ported verbatim from
    pydhn `_transition_friction_factor` (Hafsi 2021 explicit Colebrook solution).
    Only used in-band (2320 < Re < 4000); out-of-band values are masked away by
    `friction_factor` and need not be finite.
    """
    K = _safe_div(roughness, diameter * 1000.0)
    a = K / 3.7
    b = _safe_div(torch.as_tensor(2.51, dtype=Re.dtype, device=Re.device), Re)
    c = -2.0 / math.log(10.0)
    div69 = _safe_div(torch.as_tensor(6.9, dtype=Re.dtype, device=Re.device), Re)
    X0 = b * torch.abs(-1.8 * torch.log10(a ** 1.11 + div69))
    apX0 = a + X0
    c1 = b * c / (3.0 * apX0 ** 3)
    c2 = -b * c / (2.0 * apX0 ** 2) - 3.0 * c1 * X0
    c3 = 3.0 * c1 * X0 ** 2 + b * c / apX0 ** 2 * X0 + b * c / apX0 - 1.0
    c4 = (
        b * c * torch.log(apX0)
        - b * c / apX0 * X0
        - b * c / (2.0 * apX0 ** 2) * X0 ** 2
        + c1 * X0 ** 3
    )
    sigma = _safe_div(c3, 3.0 * c1) - _safe_div(c2 ** 2, 9.0 * c1 ** 2)
    phi = (
        _safe_div(c4, 2.0 * c1)
        + _safe_div(c2, 3.0 * c1) ** 3
        - _safe_div(c2 * c3, 6.0 * c1 ** 2)
    )
    x0 = _safe_div(X0, b)
    root = torch.sqrt(phi ** 2 + sigma ** 3)
    inner = (root - phi) ** (1.0 / 3.0) - (root + phi) ** (1.0 / 3.0) + (a + 3.0 * b * x0) / 2.0
    return b ** 2 * inner ** -2


def friction_factor(Re, diameter, roughness, re_floor: float = 0.0):
    """
    Darcy friction factor for all regimes (port of compute_friction_factor):
      laminar   (Re <= 2320): 64/Re
      turbulent (Re >= 4000): Haaland
      transition            : non-affine Hafsi/Colebrook

    `re_floor` (>0) clamps Re from below as defensive hygiene for the model's
    inner loop, where intermediate iterates can momentarily hit exact zero flow.
    Leave at 0.0 to match PyDHN exactly for validation.
    """
    if re_floor > 0.0:
        Re = Re.clamp_min(re_floor)
    laminar = _safe_div(torch.as_tensor(64.0, dtype=Re.dtype, device=Re.device), Re)
    turbulent = _turbulent_fd(Re, diameter, roughness)
    # Evaluate the transition formula on Re clamped to its valid band. Out-of-band
    # values are masked away by the `where` below, but clamping keeps them finite so
    # the cube-roots don't emit NaN that would poison gradients (torch.where routes
    # grads through both branches). In-band this clamp is a no-op, so forward output
    # and the Gate-1 match are unchanged.
    transition = _transition_fd(Re.clamp(2320.0, 4000.0), diameter, roughness)
    fd = torch.where(
        Re <= 2320.0,
        laminar,
        torch.where(Re >= 4000.0, turbulent, transition),
    )
    return fd


def phi(mdot, diameter, length, fd, rho: float):
    """
    Pressure drop across a pipe (friction only), port of compute_dp_pipe:
        dp = L * fd * |mdot| * mdot / (4 * pi^2 * rho * (d/2)^5)
    Sign follows the flow direction. Hydrostatic term is NOT included (it cancels
    around any closed loop, so it is irrelevant to the loop residual).
    """
    r = diameter / 2.0
    num = length * fd * mdot.abs() * mdot
    den = 4.0 * _PI ** 2 * rho * r ** 5
    return _safe_div(num, den)


def dphi_dmdot(mdot, diameter, length, fd, rho: float):
    """
    d(dp)/d(mdot) with fd held fixed w.r.t. mdot, port of compute_dp_pipe's dp_der:
        dp_der = L * fd * |mdot| / (2 * pi^2 * rho * (d/2)^5)
    This is the local hydraulic resistance used both as the NR-Jacobian diagonal
    and (recomputed per unrolled step) as the attention bias.
    """
    r = diameter / 2.0
    num = length * fd * mdot.abs()
    den = 2.0 * _PI ** 2 * rho * r ** 5
    return _safe_div(num, den)


def pipe_dp(mdot, diameter, length, roughness, rho: float, mu: float, re_floor: float = 0.0):
    """Convenience: full friction dp from flow + geometry (Re -> fd -> phi)."""
    Re = reynolds(mdot, diameter, mu)
    fd = friction_factor(Re, diameter, roughness, re_floor=re_floor)
    return phi(mdot, diameter, length, fd, rho)
