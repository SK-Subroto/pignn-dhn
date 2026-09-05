"""Ground-truth dataset generation, kept OUT of the dhn_gnn package on purpose.

This is the simulator that produces what the models are trained against; it is
not part of the model code and shares nothing with it but the network topology
on disk. It also pulls heavier, generation-only dependencies (meteostat for
weather, pydhn's thermal solver), which the training and evaluation path must
not have to import.

    python -m datagen.run --help        run a simulation
    python -m datagen.export --help     publish a run into data/solved_<version>/

Ported from the standalone `pydhnClaude/simulation` package, which replaced
`synthetic_data steady custom.ipynb` (and the earlier dhn_gnn/generate.py port
of that notebook). The notebook had three defects this implementation fixes,
documented in README.md: an unfinished `compute_producer_massflow` stub called
with the wrong arity, two mutually inconsistent reference outdoor temperatures,
and every substation pinned to one 45 degC outlet regardless of its measured
return temperature.

IMPORTANT -- this simulation is THERMOhydraulic: each pipe carries its own
temperature, so density and viscosity vary across the network. dhn_gnn's physics
currently assumes a single fixed 50 degC (config.RHO_50 / config.MU_50). That
mismatch is why `tests/run_gates.py` Gate 5 fails on this data; the run's
`edges-temperature.csv` is what a per-pipe fix would need.
"""
