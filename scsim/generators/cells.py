"""Cell parameter generation for single-cell RNA-seq simulation."""

from typing import Sequence

import numpy as np
import pandas as pd
from numpy.random import Generator


def simulate_groups(
    rng: Generator,
    ncells: int,
    ngroups: int,
    groupprob: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Sample cell group identities from a categorical distribution.

    Args:
        rng: NumPy random generator.
        ncells: Number of cells to simulate.
        ngroups: Number of cell groups.
        groupprob: Probability of a cell belonging to each group.

    Returns:
        Tuple of (group assignments array, unique groups array).
    """
    groupid = rng.choice(
        np.arange(1, ngroups + 1),
        size=ncells,
        p=groupprob,
    )
    groups = np.unique(groupid)
    return groupid, groups


def simulate_cell_params(
    rng: Generator,
    ncells: int,
    ngroups: int,
    groupprob: Sequence[float],
    libloc: float,
    libscale: float,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Sample cell group identities and library sizes.

    Args:
        rng: NumPy random generator.
        ncells: Number of cells to simulate.
        ngroups: Number of cell groups.
        groupprob: Probability of a cell belonging to each group.
        libloc: Mean of log-normal distribution for library sizes.
        libscale: Standard deviation of log-normal distribution for library sizes.

    Returns:
        Tuple of (cell parameters DataFrame, cell names list, unique groups array).
    """
    groupid, groups = simulate_groups(rng, ncells, ngroups, groupprob)
    libsize = rng.lognormal(mean=libloc, sigma=libscale, size=ncells)

    cellnames = [f"Cell{i}" for i in range(1, ncells + 1)]
    cellparams = pd.DataFrame(
        [groupid, libsize],
        index=["group", "libsize"],
        columns=cellnames,
    ).T
    cellparams["group"] = cellparams["group"].astype(int)

    return cellparams, cellnames, groups
