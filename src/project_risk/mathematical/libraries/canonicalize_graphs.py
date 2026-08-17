from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Set, List, Iterable, Any
import pickle
from collections import Counter

from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import (
    generate_connected_graphs_n_nodes,
    canonicalize_edges_with_roles)

BASE_LIB_DIR = Path("small_graph_libraries")
CANON_DIR = BASE_LIB_DIR / "canonical_topologies"



def canonical_cache_path(nA: int, nD: int, base_dir: Path = BASE_LIB_DIR) -> Path:
    d = base_dir / "canonical_topologies"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{nA}A_{nD}D.pkl"

def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def precompute_canonical_topologies(
    nA: int,
    nD: int,
    base_dir: Path = BASE_LIB_DIR,
    overwrite: bool = False,
) -> dict:
    """
    Build and save canonical representatives for (nA,nD) under A/D-preserving isomorphism.
    Returns the saved dict.
    """
    path = canonical_cache_path(nA, nD, base_dir)
    if path.exists() and not overwrite:
        return load_pickle(path)

    n = nA + nD
    labelled_graphs = generate_connected_graphs_n_nodes(n)

    canonical_reps: Dict[Tuple[Tuple[int,int], ...], Set[Tuple[int,int]]] = {}
    counts = Counter()

    for edges in labelled_graphs:
        canon_key, _p1, _p2 = canonicalize_edges_with_roles(
            edges=sorted(edges),
            num_attacker_nodes=nA,
            num_defender_nodes=nD,
        )
        counts[canon_key] += 1
        if canon_key not in canonical_reps:
            canonical_reps[canon_key] = set(canon_key)  # store canonical rep itself

    payload = {
        "nA": nA,
        "nD": nD,
        "num_nodes": n,
        "num_labelled_seen": len(labelled_graphs),
        "num_canonical": len(canonical_reps),
        "canonical_reps": canonical_reps,   # canon_key -> edges_set (rep)
        "counts": dict(counts),             # canon_key -> how many labelled map here
    }
    save_pickle(payload, path)
    return payload

def load_canonical_representatives(
    nA: int,
    nD: int,
    base_dir: Path = BASE_LIB_DIR,
) -> List[List[Tuple[int, int]]]:
    payload = load_pickle(canonical_cache_path(nA, nD, base_dir))
    reps = payload["canonical_reps"].values()
    # return as list-of-lists (stable iteration order isn’t required, but lists are convenient)
    return [sorted(list(edges_set)) for edges_set in reps]



def debug_canonicalization(
    edges,
    num_attacker_nodes: int,
    num_defender_nodes: int,
):
    """
    Debug helper to inspect A/D-preserving canonicalization.

    Prints:
      - original edges
      - canonical edges
      - permutation maps
      - attacker / defender blocks
    """
    edges = tuple(sorted(tuple(sorted(e)) for e in edges))

    canonical_edges, perm_old_to_new, perm_new_to_old = canonicalize_edges_with_roles(
        edges=edges,
        num_attacker_nodes=num_attacker_nodes,
        num_defender_nodes=num_defender_nodes,
    )

    print("\n=== CANONICALIZATION DEBUG ===")
    print(f"nA={num_attacker_nodes}, nD={num_defender_nodes}")
    print("\nOriginal edges:")
    print(edges)

    print("\nCanonical edges:")
    print(canonical_edges)

    print("\nPermutation (old → new):")
    for i, j in enumerate(perm_old_to_new):
        role = "A" if i < num_attacker_nodes else "D"
        print(f"  {i} ({role}) → {j}")

    print("\nPermutation (new → old):")
    for i, j in enumerate(perm_new_to_old):
        role = "A" if i < num_attacker_nodes else "D"
        print(f"  {i} ({role}) → {j}")

    print("\nAttacker block preserved:",
          all(i < num_attacker_nodes for i in perm_old_to_new[:num_attacker_nodes]))
    print("Defender block preserved:",
          all(i >= num_attacker_nodes for i in perm_old_to_new[num_attacker_nodes:]))

    print("==============================\n")





if __name__ == "__main__":
    for (nA,nD) in [(1,1),(2,1),(3,1),(4,1),
                    (1,2),(2,2),(3,2),
                    (1,3),(2,3),
                    (1,4)]:
        info = precompute_canonical_topologies(nA,nD, overwrite=False)
        print(f'nA nD: {nA,nD}', 
              f'Full topologies: {info["num_labelled_seen"]}', "->", 
              f'Canonicalized topologies: {info["num_canonical"]}')






