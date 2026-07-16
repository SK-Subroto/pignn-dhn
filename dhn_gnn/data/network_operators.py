"""Fixed network operators (cycle matrix B, incidence A, internal-loop mask).

Topology is constant across the dataset (Decision D1), so these are built ONCE
from the canonical PyDHN network and cached. The network is reconstructed exactly
as the v4 notebook does (opendhn-data + STEPSIZE), so edge ordering and the cycle
basis match the solver that generated the targets.

`B = net.cycle_matrix` uses PyDHN's spanning-tree basis (the one `solve_hydraulics`
actually solves with) — not the networkx basis. The internal-loop rows are
identified by replicating the masking logic in
`pydhn/solving/hydraulic_simulation.py:154-168`.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from pydhn.networks import Network
from pydhn.utilities.matrices import compute_incidence_matrix

from dhn_gnn import config


def build_network() -> Network:
    """Reconstruct the canonical v4 network (topology + geometry) from opendhn-data."""
    nodes = pd.read_csv(config.NETWORK_DIR / "nodes.csv")
    pipes = pd.read_csv(config.NETWORK_DIR / "pipes.csv")
    subs = pd.read_csv(config.NETWORK_DIR / "substations.csv")
    hs = pd.read_csv(config.NETWORK_DIR / "heating_stations.csv")
    mass_flow = pd.read_csv(config.MEASUREMENTS_DIR / "mass_flow.csv")

    net = Network()
    for _, node in nodes.iterrows():
        net.add_node(
            name=node["node_id"], x=node["x"], y=node["y"], z=node["z"],
            line="supply" if node["is_supply"] else "return",
        )
    for _, pipe in pipes.iterrows():
        net.add_pipe(
            name=pipe["pipe_id"], start_node=pipe["inlet_node"], end_node=pipe["outlet_node"],
            length=pipe["length"], diameter=pipe["d_int"], roughness=pipe["roughness"],
            internal_pipe_thickness=pipe["t_int"], insulation_thickness=pipe["t_ins"],
            casing_thickness=pipe["t_ext"], k_insulation=pipe["lambda_ins"],
            line="supply" if pipe["is_supply"] else "return",
            depth=0 if pipe["is_aerial"] else 0.8,
            stepsize=config.STEPSIZE, discretization=1,
        )
    for idx, h in hs.iterrows():
        if idx == 0:
            net.add_producer(
                name=h["hs_id"], start_node=h["inlet_node"], end_node=h["outlet_node"],
                setpoint_type_hx="t_out", setpoint_type_hyd="pressure",
                setpoint_value_hyd=-100000, stepsize=config.STEPSIZE,
            )
        else:
            net.add_producer(
                name=h["hs_id"], start_node=h["inlet_node"], end_node=h["outlet_node"],
                setpoint_type_hx="t_out", setpoint_type_hyd="mass_flow",
                setpoint_value_hyd=mass_flow["HS1"].iloc[0], stepsize=config.STEPSIZE,
            )
    for _, sub in subs.iterrows():
        net.add_consumer(
            name=sub["sub_id"], start_node=sub["inlet_node"], end_node=sub["outlet_node"],
            setpoint_type_hx="t_out", setpoint_value_hx=45,
            setpoint_type_hyd="mass_flow", control_type="mass_flow",
            stepsize=config.STEPSIZE,
        )
    return net


def _internal_loop_rows(net, B) -> np.ndarray:
    """
    Rows of B that are true internal mesh loops (exclude pressure- and mass-flow-
    setpoint-carrying loops). Verbatim port of hydraulic_simulation.py:154-168.
    """
    leaves = net.leaf_components_mask
    p_mask = np.intersect1d(leaves, net.pressure_setpoints_mask)
    m_mask = np.intersect1d(leaves, net.mass_flow_setpoints_mask)
    main_idx = p_mask[0]
    secondary = np.setdiff1d(leaves, main_idx)
    matrix_indices = np.arange(B.shape[0])
    matrix_indices_p = np.where(np.in1d(secondary, p_mask))[0]
    matrix_indices_m = np.where(np.in1d(secondary, m_mask))[0]
    loops = np.setdiff1d(matrix_indices, matrix_indices_p)
    loops = np.setdiff1d(loops, matrix_indices_m)
    return loops


@dataclass
class NetworkOperators:
    """Cached, fixed operators for the whole dataset. All tensors are float64."""
    B: torch.Tensor           # (L, E) cycle matrix, entries in {-1,0,1}
    A: torch.Tensor           # (N, E) oriented node-edge incidence
    internal_loops: torch.Tensor   # (L_int,) long, row indices of B
    edge_names: list          # length E, canonical edge order (== CSV columns)
    node_names: list          # length N
    pipe_mask: torch.Tensor   # (E,) bool, True for base_pipe edges
    diameter: torch.Tensor    # (E,) m   (0 for non-pipe edges)
    length: torch.Tensor      # (E,) m
    roughness: torch.Tensor   # (E,) mm

    @property
    def n_edges(self):
        return self.B.shape[1]

    @property
    def n_loops(self):
        return self.B.shape[0]


def build_operators(dtype=torch.float64) -> NetworkOperators:
    """Build the network once and extract all fixed operators."""
    net = build_network()

    B = np.asarray(net.cycle_matrix, dtype=np.float64)
    A = compute_incidence_matrix(net, oriented=True).astype(np.float64)
    internal = _internal_loop_rows(net, B)

    edge_arr, edge_names = net.edges(data="name")
    edge_names = list(edge_names)
    node_names = list(list(net.nodes())[0])

    # Per-edge geometry (0 for non-pipe edges; pipes identified by component_type)
    _, diameter = net.edges(data="diameter")
    _, length = net.edges(data="length")
    _, roughness = net.edges(data="roughness")
    comp_type = net.get_edges_attribute_array("component_type")
    pipe_mask = comp_type == "base_pipe"

    def _num(a):
        return torch.as_tensor(np.nan_to_num(np.asarray(a, dtype=np.float64)), dtype=dtype)

    return NetworkOperators(
        B=torch.as_tensor(B, dtype=dtype),
        A=torch.as_tensor(A, dtype=dtype),
        internal_loops=torch.as_tensor(internal, dtype=torch.long),
        edge_names=edge_names,
        node_names=node_names,
        pipe_mask=torch.as_tensor(pipe_mask, dtype=torch.bool),
        diameter=_num(diameter),
        length=_num(length),
        roughness=_num(roughness),
    )
