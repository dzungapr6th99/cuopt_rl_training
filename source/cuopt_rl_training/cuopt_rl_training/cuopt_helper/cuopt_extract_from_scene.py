from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional

from pxr import Usd, UsdGeom

_NODE_RE_CACHE: Dict[str, re.Pattern] = {}

def extract_nodes_from_stage(
    stage: Usd.Stage,
    root_prim_path: Optional[str] = None,
    node_prefix: str = "node_",
) -> List[Tuple[float, float, float]]:
    """
    Find prims whose name matches f"{node_prefix}<int>" and return a list:
      node_xyz[i] = (x,y,z) in WORLD coordinates

    Requirements:
      - nodes must be contiguous indices from 0..N-1, otherwise raises.
    """
    key = node_prefix.lower()
    if key not in _NODE_RE_CACHE:
        _NODE_RE_CACHE[key] = re.compile(rf"^{re.escape(node_prefix)}(\d+)$", re.IGNORECASE)
    node_re = _NODE_RE_CACHE[key]

    if root_prim_path:
        root = stage.GetPrimAtPath(root_prim_path)
        if not root.IsValid():
            raise ValueError(f"Invalid root prim path: {root_prim_path}")
        prim_iter = Usd.PrimRange(root)
    else:
        prim_iter = stage.Traverse()

    nodes: Dict[int, Tuple[float, float, float]] = {}
    for prim in prim_iter:
        m = node_re.match(prim.GetName())
        if not m:
            continue
        idx = int(m.group(1))
        if not prim.IsValid():
            continue

        xform = UsdGeom.Xformable(prim)
        tf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetOrthonormalized()
        t = tf.ExtractTranslation()
        nodes[idx] = (float(t[0]), float(t[1]), float(t[2]))

    if not nodes:
        raise RuntimeError(f"No nodes found with prefix '{node_prefix}' under {root_prim_path or 'stage'}.")

    max_idx = max(nodes.keys())
    missing = [i for i in range(max_idx + 1) if i not in nodes]
    if missing:
        raise RuntimeError(f"Missing node indices: {missing[:20]}{'...' if len(missing)>20 else ''}")
    return [nodes[i] for i in range(max_idx + 1)]

_EDGE_RE_CACHE: Dict[str, re.Pattern] = {}

def extract_edges_from_stage(
    stage: Usd.Stage,
    n_nodes: int,
    root_prim_path: Optional[str] = None,
    edge_prefix: str = "edge_",
    bidirectional: bool = True,
) -> List[Tuple[int, int]]:
    """
    Find prims whose name matches f"{edge_prefix}<u>_<v>" and return edges list (u,v).
    """
    key = edge_prefix.lower()
    if key not in _EDGE_RE_CACHE:
        _EDGE_RE_CACHE[key] = re.compile(rf"^{re.escape(edge_prefix)}(\d+)_(\d+)$", re.IGNORECASE)
    edge_re = _EDGE_RE_CACHE[key]

    if root_prim_path:
        root = stage.GetPrimAtPath(root_prim_path)
        if not root.IsValid():
            raise ValueError(f"Invalid root prim path: {root_prim_path}")
        prim_iter = Usd.PrimRange(root)
    else:
        prim_iter = stage.Traverse()

    edges: List[Tuple[int, int]] = []
    for prim in prim_iter:
        m = edge_re.match(prim.GetName())
        if not m:
            continue
        u = int(m.group(1))
        v = int(m.group(2))
        if not (0 <= u < n_nodes and 0 <= v < n_nodes):
            raise RuntimeError(f"Edge '{prim.GetName()}' references node out of range (0..{n_nodes-1}).")
        edges.append((u, v))
        if bidirectional:
            edges.append((v, u))

    if not edges:
        raise RuntimeError(f"No edges found with prefix '{edge_prefix}' under {root_prim_path or 'stage'}.")

    return edges
