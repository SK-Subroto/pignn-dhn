"""Project-wide constants and dataset paths for the physics-core spike.

The canonical dataset is `gen_data_steady_v4` (Decision in spike_build_plan.md):
a hydraulics-only run (`SimpleStep(with_thermal=False)`) with all substations
mass-flow-controlled and friction evaluated isothermally at PyDHN's default 50 degC.
That isothermal choice is why the fluid properties below are exact constants.
"""

import os
from pathlib import Path

import numpy as np

# --- Fluid properties: exact PyDHN Water() at the v4 fixed temperature (50 degC) ---
# Reproduce with: from pydhn.fluids import Water; Water().get_rho(50), Water().get_mu(50)
FIXED_TEMPERATURE_C = 50.0
RHO_50 = 988.0038459097711      # kg/m^3
MU_50 = 5.466190912183895e-04   # Pa.s

# --- Network construction (must match the v4 notebook) ---
STEPSIZE = 3600  # seconds per step; passed to add_pipe/add_producer/add_consumer

# --- Data location ------------------------------------------------------------
# Data is bundled inside the project at pignn-dhn/data/ (copied from the original
# pydhn-simulations repo, canonical dataset = steady_v4). Override the root with
# the DHN_DATA_ROOT environment variable if you relocate it.
DATA_ROOT = Path(
    os.environ.get(
        "DHN_DATA_ROOT",
        str(Path(__file__).resolve().parents[1] / "data"),
    )
)

NETWORK_DIR = DATA_ROOT / "network"           # topology + pipe geometry
MEASUREMENTS_DIR = DATA_ROOT / "measurements"  # real boundary snapshot
GEN_DATA_DIR = DATA_ROOT / "solved_steady_v4"  # PyDHN ground-truth outputs

# Solved-state CSVs used by the integration gates
MASS_FLOW_CSV = GEN_DATA_DIR / "edges-mass_flow.csv"
DELTA_P_CSV = GEN_DATA_DIR / "edges-delta_p.csv"
DELTA_P_FRICTION_CSV = GEN_DATA_DIR / "edges-delta_p_friction.csv"
NODE_PRESSURE_CSV = GEN_DATA_DIR / "nodes-pressure.csv"

# --- Outputs -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT = RESULTS_DIR / "model.pt"

# --- Model / solver defaults ---------------------------------------------------
# K is shared by train.py and evaluate.py ON PURPOSE: the solver carries one
# nn.Linear head PER unrolled step, so a checkpoint trained at K steps cannot be
# loaded into a model built with a different K. Changing K here changes both.
# Full Newton solves the exact 12x12 loop system, so it converges quadratically and
# needs no damping: it passes PyDHN's own 50 Pa mark in ~4 steps and reaches 1e-4 Pa
# by step 20, where damped diagonal Newton is still at ~30 Pa. K=10 leaves margin.
K_UNROLLED = 10

MODEL_KWARGS = dict(
    K=K_UNROLLED,
    d_model=64,
    n_heads=4,
    num_attn_layers=2,
    step_scale=0.5,
    newton_damping=1.0,
    newton_mode="full",
)

# Early-exit tolerance for evaluation: stop unrolling once the internal-loop
# residual is below this. Set to PyDHN's own convergence threshold (50 Pa), which
# is the level at which the reference solution itself was declared converged.
# NOT the 15 Pa figure quoted in train.py: the solver bottoms out around 24 Pa on
# this network, so a 15 Pa tolerance never fires and the early exit is dead code.
EVAL_TOL_PA = 50.0

# Learned-initializer architecture (dhn_gnn.model.initializer): one attention pass
# predicts the loop-space solution, then a few exact Newton steps polish it.
INIT_MODEL_KWARGS = dict(
    n_newton=4,
    d_model=64,
    n_heads=4,
    num_attn_layers=2,
)
CHECKPOINT_INIT = RESULTS_DIR / "model_init.pt"


def get_device(pref: str = "auto"):
    """
    Resolve the compute device. 'auto' picks CUDA when available, else CPU.

    A caution specific to this model, measured on this network: at batch size 1 the
    per-step tensors are tiny (1352 nodes, d_model 64) and the unroll is sequential,
    so CUDA kernel-launch overhead dominates and the GPU is NOT reliably faster than
    CPU -- it can be slower. The GPU only pays off once many timesteps are solved in
    one batched pass, which the learned-initializer architecture makes possible
    (timesteps are independent) and warm-starting does not.
    """
    import torch
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable. Check: torch.__version__ must be a "
                "+cu build (a +cpu build never sees a GPU), the NVIDIA driver must "
                "be recent enough for that CUDA version, and the GPU's compute "
                "capability must appear in torch.cuda.get_arch_list()."
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Train / test split --------------------------------------------------------
def split_timesteps(n_timesteps: int, test_every: int = 5):
    """
    Interleaved split of the simulation window into (train_ts, test_ts).

    Interleaved, NOT contiguous: the dataset is driven by a month of real weather,
    so a contiguous tail would be a colder/warmer regime than the head and the
    "test" score would measure distribution shift rather than generalization.
    Every `test_every`-th timestep is held out.
    """
    ts = np.arange(n_timesteps)
    test = ts[ts % test_every == 0]
    train = ts[ts % test_every != 0]
    return train, test
