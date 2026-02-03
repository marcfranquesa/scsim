"""Doublet simulation for single-cell RNA-seq."""

import numpy as np
import pandas as pd
from numpy.random import Generator


def simulate_doublets(
    rng: Generator,
    cellparams: pd.DataFrame,
    cellgenemean: pd.DataFrame,
    cellnames: list[str],
    ncells: int,
    ndoublets: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Simulate doublet cells by merging expression profiles.

    Doublets are created by combining expression profiles from
    two cells while preserving the total library size.

    Args:
        rng: NumPy random generator.
        cellparams: Cell parameters DataFrame (modified in place).
        cellgenemean: Cell-gene mean expression DataFrame (modified in place).
        cellnames: List of cell names.
        ncells: Number of non-doublet cells.
        ndoublets: Number of doublet cells to simulate.

    Returns:
        Tuple of (updated cellparams, updated cellgenemean, updated cellnames).
    """
    # Select doublet cells and determine the second cell to merge with
    d_ind = sorted(rng.choice(ncells, ndoublets, replace=False))
    d_ind = [f"Cell{x + 1}" for x in d_ind]

    cellparams["is_doublet"] = False
    cellparams.loc[d_ind, "is_doublet"] = True

    extraind = cellparams.index[-ndoublets:]
    group2 = cellparams.loc[extraind, "group"].values
    cellparams["group2"] = -1
    cellparams.loc[d_ind, "group2"] = group2

    # Update the cell-gene means for the doublets while preserving
    # the same library size
    dmean = cellgenemean.loc[d_ind, :].values
    dmultiplier = 0.5 / dmean.sum(axis=1)
    dmean = np.multiply(dmean, dmultiplier[:, np.newaxis])

    omean = cellgenemean.loc[extraind, :].values
    omultiplier = 0.5 / omean.sum(axis=1)
    omean = np.multiply(omean, omultiplier[:, np.newaxis])

    newmean = dmean + omean
    libsize = cellparams.loc[d_ind, "libsize"].values
    newmean = np.multiply(newmean, libsize[:, np.newaxis])
    cellgenemean.loc[d_ind, :] = newmean

    # Remove extra doublet cells from the data structures
    cellgenemean.drop(extraind, axis=0, inplace=True)
    cellparams.drop(extraind, axis=0, inplace=True)
    cellnames = cellnames[:ncells]

    return cellparams, cellgenemean, cellnames
