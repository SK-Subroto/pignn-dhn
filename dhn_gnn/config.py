"""Project-wide constants and dataset paths for the physics-core spike.

The canonical dataset is `gen_data_steady_v4` (Decision in spike_build_plan.md):
a hydraulics-only run (`SimpleStep(with_thermal=False)`) with all substations
mass-flow-controlled and friction evaluated isothermally at PyDHN's default 50 degC.
That isothermal choice is why the fluid properties below are exact constants.
"""

import os
from pathlib import Path

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
