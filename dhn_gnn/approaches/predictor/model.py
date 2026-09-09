"""Pure predictor: one GNN pass IS the answer. No Newton, no iteration.

This is the ablation that isolates the learned component. The initializer proved
that one attention pass lands close enough for ~1.6 exact Newton steps to finish;
this asks the harder question -- can the network get all the way there alone?

The measured starting point, from results/report_data.json: the initializer's raw
guess (before any Newton step) sits at a median 277 Pa, against a 50 Pa tolerance.
So a pure predictor must be about 5.5x more accurate than the network already is.
Near the solution the residual is roughly linear in the loop-flow error, so that
is a 5.5x demand on the prediction itself -- not obviously reachable, which is
exactly why it is worth measuring rather than assuming.

Architecture is DELIBERATELY identical to the initializer's, with n_newton pinned
to 0. That is what makes the comparison clean: any difference in the result is the
Newton polish, not a different network. The trainer, however, is not the same --
see train.py, which defaults to optimizing the physics residual rather than
regressing a_star, because with no polish behind it the residual IS the metric.
"""

from dhn_gnn import config
from dhn_gnn.approaches.initializer.model import DHNInitializerSolver


class DHNPredictorSolver(DHNInitializerSolver):
    """The initializer's network with the Newton polish removed entirely."""

    def __init__(self, ops, d_model: int = 64, n_heads: int = 4,
                 num_attn_layers: int = 2, node_scope: str = "all",
                 cycle_hops: int = 1,
                 rho: float = config.RHO_50, mu: float = config.MU_50):
        # n_newton=0 makes NewtonSolver.forward return the guess untouched, with
        # residual_terms == [residual(guess)]. newton_mode is then never reached,
        # so it is not exposed as a knob -- an inert hyperparameter in a saved
        # config is worse than no hyperparameter, because it reads as a choice.
        super().__init__(ops, n_newton=0, d_model=d_model, n_heads=n_heads,
                         num_attn_layers=num_attn_layers, newton_mode="full",
                         node_scope=node_scope, cycle_hops=cycle_hops,
                         rho=rho, mu=mu)

    def forward(self, mdot0, tol=None, n_newton=None, c_init=None):
        """
        Identical signature to NewtonSolver so every report can treat this like
        any other solver, but `n_newton` and `tol` are ignored by construction:
        there is exactly one step and nothing to early-exit from. Accepting and
        silently honouring an n_newton here would quietly turn the ablation back
        into the initializer.
        """
        return super().forward(mdot0, tol=None, n_newton=0, c_init=c_init)
