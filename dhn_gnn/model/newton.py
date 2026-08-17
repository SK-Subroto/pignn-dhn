"""Pure-physics Newton solver -- the control condition, with zero learned parameters.

This is the baseline every learned variant has to beat, so it deserves to be a real
class rather than "a network with its output layer zeroed out", which is how it
previously existed. Two things follow from making it explicit:

  * it can be run and cited on its own, without loading a checkpoint;
  * the measurement that matters most stays honest. On this network, exact Newton
    from a cold start reaches PyDHN's 50 Pa tolerance in 3.69 steps and is ~9x
    faster than the original GNN-every-step architecture -- with no machine
    learning involved at all. Any learned component must be compared against
    THIS, not against the original model.

`c_init` is the only knob: leave it at zero for a cold start, or pass the previous
timestep's solution for the warm start that PyDHN uses (which is what forces PyDHN
to solve timesteps strictly in order).
"""

import torch

from dhn_gnn import config
from dhn_gnn.model.base import DHNPhysicsBase


class NewtonSolver(DHNPhysicsBase):
    def __init__(self, ops, n_newton: int = 4, newton_mode: str = "full",
                 rho: float = config.RHO_50, mu: float = config.MU_50):
        super().__init__(ops, rho=rho, mu=mu)
        self.n_newton = n_newton
        self.newton_mode = newton_mode

    def initial_guess(self, mdot0):
        """Cold start. Subclasses override this -- it is the ONLY learned part."""
        return mdot0.new_zeros(self.n_free)

    def forward(self, mdot0, tol=None, n_newton=None, c_init=None):
        """
        mdot0: (E,) boundary-feasible reference flow.
        tol:   stop once max|residual| < tol (Pa). None runs the full count.
        Returns (mdot, residual_terms, c_hist).

        residual_terms[0] is the residual of the STARTING POINT, before any Newton
        step. Keeping it in the list is what lets the reports separate "how good was
        the guess" from "how much work did Newton do" -- but it also means
        len(residual_terms) is one MORE than the number of Newton steps taken.
        """
        n = self.n_newton if n_newton is None else n_newton
        c = self.initial_guess(mdot0) if c_init is None else c_init

        residual_terms = [self.residual(self.edge_flow(mdot0, c))]
        c_hist = [c]

        for _ in range(n):
            if tol is not None and residual_terms[-1].abs().max() < tol:
                break
            mdot = self.edge_flow(mdot0, c)
            c = c - self.newton_step(self.residual(mdot), self.dp_der(mdot),
                                     self.newton_mode)
            c_hist.append(c)
            residual_terms.append(self.residual(self.edge_flow(mdot0, c)))

        return self.edge_flow(mdot0, c), residual_terms, c_hist
