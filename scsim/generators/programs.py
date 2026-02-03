"""Expression program simulation for single-cell RNA-seq."""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from numpy.random import Generator


def simulate_program(
    rng: Generator,
    geneparams: pd.DataFrame,
    cellparams: pd.DataFrame,
    nproggenes: int,
    ngenes: int,
    ngroups: int,
    progdownprob: float,
    progdeloc: float,
    progdescale: float,
    proggroups: Optional[Sequence[int]],
    progcellfrac: float,
    minprogusage: float,
    maxprogusage: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a shared gene expression program.

    Creates a program affecting a subset of genes that is active
    in a fraction of cells with varying usage levels.

    Args:
        rng: NumPy random generator.
        geneparams: Gene parameters DataFrame (modified in place).
        cellparams: Cell parameters DataFrame (modified in place).
        nproggenes: Number of genes in the expression program.
        ngenes: Total number of genes.
        ngroups: Number of cell groups.
        progdownprob: Probability that a program gene is downregulated.
        progdeloc: Mean of log-normal distribution for program DE.
        progdescale: Standard deviation of log-normal distribution for program DE.
        proggroups: Which cell groups can have the program active.
        progcellfrac: Fraction of cells in eligible groups with active program.
        minprogusage: Minimum program usage level.
        maxprogusage: Maximum program usage level.

    Returns:
        Tuple of (updated geneparams, updated cellparams).
    """
    # Simulate the program gene expression
    geneparams["prog_gene"] = False
    proggenes = geneparams.index[-nproggenes:]
    geneparams.loc[proggenes, "prog_gene"] = True

    de_ratio = rng.lognormal(mean=progdeloc, sigma=progdescale, size=nproggenes)
    de_ratio[de_ratio < 1] = 1 / de_ratio[de_ratio < 1]

    is_downregulated = rng.choice(
        [True, False],
        size=len(de_ratio),
        p=[progdownprob, 1 - progdownprob],
    )
    de_ratio[is_downregulated] = 1.0 / de_ratio[is_downregulated]

    all_de_ratio = np.ones(ngenes)
    all_de_ratio[-nproggenes:] = de_ratio
    prog_mean = geneparams["gene_mean"] * all_de_ratio
    geneparams["prog_genemean"] = prog_mean

    # Assign the program to cells
    cellparams["has_program"] = False
    if proggroups is None:
        # The program is active in all cell types
        proggroups = np.arange(1, ngroups + 1)

    cellparams.loc[:, "program_usage"] = 0.0  # Use float to avoid dtype warning

    for g in proggroups:
        groupcells = cellparams.index[cellparams["group"] == g]
        hasprog = rng.choice(
            [True, False],
            size=len(groupcells),
            p=[progcellfrac, 1 - progcellfrac],
        )
        cellparams.loc[groupcells[hasprog], "has_program"] = True
        usages = rng.uniform(
            low=minprogusage,
            high=maxprogusage,
            size=len(groupcells[hasprog]),
        )
        cellparams.loc[groupcells[hasprog], "program_usage"] = usages

    return geneparams, cellparams
